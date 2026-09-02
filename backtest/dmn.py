# -*- coding: utf-8 -*-
"""深度动量网络 (Deep Momentum Networks, Lim/Zohren/Roberts 2019) 落地 + 与慢动量 v2 对打。

核心命题：
- 慢动量 v2 = 传统时间序列动量的手工版：手工定回看 30 天 (mom_720) + 手工定门控偏移 1.0。
- DMN = 它的深度学习升级：把「手工选回看窗口 + 手工设阈值」交给 LSTM 从收益序列端到端学出来。
- 论文证据：88 个连续期货 1990-2015，Sharpe 优化的 LSTM 比传统 TSM 提升 >2 倍（无成本），
  扣 2-3bp 成本仍胜；网络自动学会了「趋势 + 波动率缩放」。

本模块做什么（walk-forward，无前视，与 replay.py 同纪律）：
1. 1h K线 → 日线收益（动量半衰期周级，日频是论文同款频率）。
2. 输入 = 过去 90 天日收益（除以自身滚动 std = 波动率缩放）；目标 = 未来 96h 收益。
3. LSTM → tanh 输出「DMN 分」∈[-1,1]；损失 = 负 Sharpe（横截面组合，论文同款）。
4. 前 1 年训练，后 1 年测试。测试期用 DMN 分替代 mom_720 走同一回放管线
   （趋势门控 + top20 + 3×ATR 止损 + 5% 仓位），与 v2 同窗口对打。

用法：
    python backtest/dmn.py [--epochs 40] [--save]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backtest.replay import (
    load_klines, prep_features, replay_momentum, _max_dd,
    MOM_TOP_N, MOM_POS_MAX, MOM_TREND_OFFSET, ATR_MULT,
)

OUT = Path("data/model_out")
SEQ_LEN = 90            # 输入窗口：过去 90 天日收益（网络可在此内学回看窗口）
HIDDEN = 64             # LSTM 隐层
LAYERS = 1
TARGET_H = 96           # 目标：未来 96h 收益（与系统持有期一致）


class DMN(nn.Module):
    """LSTM → 线性 → tanh，输出仓位分数 ∈[-1,1]。所有币共享同一网络（论文同款）。"""

    def __init__(self, input_size: int = 1, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                # (N, SEQ, hidden)
        last = out[:, -1, :]                 # (N, hidden)
        return torch.tanh(self.head(last)).squeeze(-1)   # (N,)


def build_samples(df: pd.DataFrame) -> tuple[list, list]:
    """从 1h 特征 df 提取日线样本。返回 (samples, times) 两组 list。

    每个 sample = {ts, symbol, win[SEQ_LEN], tgt}；win 是波动率缩放的过去 90 天日收益，
    tgt 是未来 96h 收益（f_close/close-1 的对数）。
    """
    daily = df[df["open_time"] % 86400 == 0].sort_values(["symbol", "open_time"])
    daily["dr"] = daily.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))
    daily["tgt"] = np.log(daily["f_close"] / daily["close"])

    samples = []
    for sym, g in daily.groupby("symbol"):
        g = g.sort_values("open_time")
        dr = g["dr"].values
        tgt = g["tgt"].values
        ts = g["open_time"].values
        # 波动率缩放：滚动 90 天 std（shift 1 = 只用 t 之前，无前视）
        dr_std = pd.Series(dr).rolling(SEQ_LEN, min_periods=30).std().shift(1).values
        for i in range(SEQ_LEN, len(g)):
            if not np.isfinite(tgt[i]):
                continue
            win = dr[i - SEQ_LEN:i] / dr_std[i - SEQ_LEN:i]
            win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(win).all():
                continue
            # 目标也做波动率缩放（论文持仓 = score/vol，等价于预测 vol 归一化收益），
            # 避免高波动币的原始收益主导损失。
            v = dr_std[i] if np.isfinite(dr_std[i]) and dr_std[i] > 0 else np.nan
            if not np.isfinite(v):
                continue
            samples.append({"ts": int(ts[i]), "symbol": sym, "win": win,
                            "tgt": float(tgt[i] / v), "tgt_raw": float(tgt[i])})

    times = sorted({s["ts"] for s in samples})
    return samples, times


def to_tensors(samples: list, device: torch.device = torch.device("cpu")):
    X = torch.tensor(np.stack([s["win"] for s in samples]), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor([s["tgt"] for s in samples], dtype=torch.float32)
    return X.to(device), y.to(device)


def train_model(X: torch.Tensor, y: torch.Tensor, day_ids: torch.Tensor,
                epochs: int = 40, lr: float = 1e-3, seed: int = 42) -> DMN:
    """按负 Sharpe 训练（横截面组合：每天 = 各币 score×tgt 等权均值）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DMN()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    D = int(day_ids.max().item()) + 1
    ones = torch.ones_like(y)

    def sharpe():
        s = model(X)
        port = s * y
        ssum = torch.zeros(D).scatter_add_(0, day_ids, port)
        scnt = torch.zeros(D).scatter_add_(0, day_ids, ones)
        pday = ssum / scnt.clamp(min=1)
        return pday.mean() / (pday.std() + 1e-8)

    for ep in range(epochs):
        opt.zero_grad()
        loss = -sharpe()
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                sh = sharpe().item()
            print(f"    epoch {ep:3d}/{epochs}  train_sharpe={sh:+.3f}")
    return model


def score_samples(model: DMN, samples: list, device: torch.device = torch.device("cpu")) -> np.ndarray:
    X, _ = to_tensors(samples, device)
    model.eval()
    with torch.no_grad():
        s = model(X).cpu().numpy()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    print("加载 1h K线 + 预计算因子 ...")
    df = load_klines()
    df, _, _ = prep_features(df)
    print(f"  {len(df):,} 条, {df['symbol'].nunique()} 币")

    print("构建日线样本（过去 90 天收益 → 未来 96h 收益）...")
    samples, times = build_samples(df)
    print(f"  {len(samples):,} 个样本, {len(times)} 个日截面")

    # —— walk-forward：前 1 年训练，后 1 年测试 ——
    mid = times[len(times) // 2]
    train = [s for s in samples if s["ts"] < mid]
    test = [s for s in samples if s["ts"] >= mid]
    print(f"  训练 {len(train):,}（~前1年）/ 测试 {len(test):,}（~后1年），分割点 {pd.to_datetime(mid, unit='s').date()}")

    Xtr, ytr = to_tensors(train)
    # 训练样本的「日」id，用于按天聚合算组合 Sharpe
    tr_ts = np.array([s["ts"] for s in train])
    day_map = {t: i for i, t in enumerate(sorted(set(tr_ts.tolist())))}
    day_ids = torch.tensor([day_map[t] for t in tr_ts], dtype=torch.long)

    print(f"训练 LSTM（{a.epochs} epochs, 负 Sharpe 损失）...")
    model = train_model(Xtr, ytr, day_ids, epochs=a.epochs)

    # —— 测试期评分 + IC 抽查 ——
    score = score_samples(model, test)
    ts_test = np.array([s["ts"] for s in test])
    tgts_raw = np.array([s["tgt_raw"] for s in test])
    tgts_vol = np.array([s["tgt"] for s in test])

    def _ic(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return 0.0
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    ic_dmn_raw = _ic(score, tgts_raw)
    ic_dmn_vol = _ic(score, tgts_vol)

    # 对照：手写 30 天动量 mom_720 在同一样本外的 IC（同样的 (symbol, ts) 对齐）
    daily_mom = df[df["open_time"] % 86400 == 0].set_index(["symbol", "open_time"])["mom_720"]
    mom_map = daily_mom.to_dict()
    mom_vals = np.array([mom_map.get((s["symbol"], s["ts"]), np.nan) for s in test])
    ic_mom_raw = _ic(mom_vals, tgts_raw)
    ic_mom_vol = _ic(mom_vals, tgts_vol)

    print(f"\n样本外 IC（对未来 96h 收益的秩相关，>0 才有预测力）:")
    print(f"  DMN·LSTM     原始收益 IC={ic_dmn_raw:+.4f} | 波动率缩放 IC={ic_dmn_vol:+.4f}")
    print(f"  手写 mom_720  原始收益 IC={ic_mom_raw:+.4f} | 波动率缩放 IC={ic_mom_vol:+.4f}")

    # 把 DMN 分并回 df（只在测试期日截面有值），走 replay 同管线
    df["dmn_score"] = np.nan
    idx = {s["symbol"]: {} for s in test}
    for s, sc in zip(test, score):
        idx[s["symbol"]][s["ts"]] = float(sc)
    # 用 merge 高效回填
    dmn_map = {}
    for s, sc in zip(test, score):
        dmn_map[(s["symbol"], s["ts"])] = float(sc)
    df["dmn_score"] = [dmn_map.get((sym, ts), np.nan) for sym, ts in zip(df["symbol"], df["open_time"])]

    start_ts = int(ts_test.min())
    end_ts = int(ts_test.max())

    print("\n" + "=" * 70)
    print(f"测试期对打（{pd.to_datetime(start_ts, unit='s').date()} ~ {pd.to_datetime(end_ts, unit='s').date()}）")
    print("同一管线：趋势门控 z-1.0 + top20 + 3×ATR 止损 + 每币 5%，唯一差异 = 排序分")
    print("=" * 70)

    r_dmn = replay_momentum(df, lookback_h=720, top_n=MOM_TOP_N, offset=MOM_TREND_OFFSET,
                            pos_max=MOM_POS_MAX, start_ts=start_ts, end_ts=end_ts,
                            score_col="dmn_score")
    r_v2 = replay_momentum(df, lookback_h=720, top_n=MOM_TOP_N, offset=MOM_TREND_OFFSET,
                           pos_max=MOM_POS_MAX, start_ts=start_ts, end_ts=end_ts)

    def fmt(name, st):
        return (f"  {name:<16} 累计 {st['total_ret']*100:+6.1f}% | 回撤 {st['max_drawdown']*100:5.1f}% | "
                f"命中 {st['hit_rate']*100:4.1f}% | {st['n_trades']}笔/{st['n_cross']}截面")

    print(fmt("DMN·LSTM", r_dmn["stats"]))
    print(fmt("慢动量 v2", r_v2["stats"]))
    print("=" * 70)

    if a.save:
        OUT.mkdir(exist_ok=True)
        payload = {
            "name": "dmn_vs_v2",
            "test_start": pd.to_datetime(start_ts, unit="s").date().isoformat(),
            "test_end": pd.to_datetime(end_ts, unit="s").date().isoformat(),
            "dmn_ic_raw": ic_dmn_raw,
            "dmn_ic_vol": ic_dmn_vol,
            "mom_ic_raw": ic_mom_raw,
            "mom_ic_vol": ic_mom_vol,
            "arch": {"seq_len": SEQ_LEN, "hidden": HIDDEN, "layers": LAYERS, "epochs": a.epochs},
            "dmn": {"equity": r_dmn["equity"], "stats": r_dmn["stats"]},
            "v2": {"equity": r_v2["equity"], "stats": r_v2["stats"]},
        }
        (OUT / "dmn_vs_v2.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
        print(f"\n已落盘 data/model_out/dmn_vs_v2.json（DMN IC={ic_dmn_vol:+.4f} vs mom_720 IC={ic_mom_vol:+.4f}）")

    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
