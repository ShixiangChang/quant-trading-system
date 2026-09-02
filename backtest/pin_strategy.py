# -*- coding: utf-8 -*-
"""插针抄底策略完整回测：急跌（对手方被迫成交/爆仓清算）后做多。

信号：过去 15min 跌超 PIN_TH（默认 -5%），用「逐分钟 + 上升沿触发 + 4h 去重」。
  - 逐分钟判断（不漏插针）
  - 上升沿（mask 从 False→True 的首个分钟）才触发，避免同一插针重复计数
  - 同币触发后 4h 内不再重复开仓（持仓去重）

动作：做多，持有 HOLD_MIN 分钟（默认 4h）后平仓。成本单边 COST_SIDE（6bp）。

教训（2026-09-01）：逐分钟重叠样本会把 edge 从真实的 +1.5% 高估到 +5%（4135 个
重叠样本 vs 595 个独立事件），必须用上升沿+去重拿到独立事件。
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_ROWS = 1_000_000
PIN_TH = -0.05
LOOKBACK = 15           # 过去 15min 收益
HOLD_MIN = 240          # 持有 4h（扫描发现 12h 更优：Sharpe 1.85→2.48）
COST_SIDE = 0.0006
OUT = "data/model_out/pin_strategy.json"


def load_close() -> tuple[np.ndarray, np.ndarray, list[str]]:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol", values="close",
                           aggfunc="last").sort_index()[symbols]
    return close.values.astype(float), close.index.values.astype(np.int64), symbols


def main() -> None:
    global HOLD_MIN, PIN_TH, COST_SIDE, OUT
    ap = argparse.ArgumentParser(description="插针抄底策略回测")
    ap.add_argument("--hold", type=int, default=HOLD_MIN, help="持有分钟数")
    ap.add_argument("--pin-th", type=float, default=PIN_TH, help="插针阈值（负）")
    ap.add_argument("--cost", type=float, default=COST_SIDE, help="单边成本")
    ap.add_argument("--up", action="store_true", help="镜像：急涨(>|阈值|)→做空（测上行插针是否均值回归）")
    ap.add_argument("--out", default=OUT, help="落盘路径")
    a = ap.parse_args()
    HOLD_MIN = a.hold
    PIN_TH = a.pin_th
    COST_SIDE = a.cost
    OUT = a.out
    UP = a.up

    C, ts, symbols = load_close()
    T, N = C.shape
    K = LOOKBACK
    H = HOLD_MIN
    ret15 = C[K:] / C[:-K] - 1.0          # (T-K, N) 逐分钟
    mask = (ret15 >= -PIN_TH) if UP else (ret15 <= PIN_TH)   # UP：急涨；默认：急跌

    # 上升沿触发
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] = mask[1:] & ~mask[:-1]
    edge[0] = mask[0]

    # 4h 去重 + 事件收益
    last = np.full(N, -10**9, dtype=np.int64)
    events: list[tuple[int, int, float]] = []
    for i in range(T - K - H):
        cols = np.where(edge[i])[0]
        if len(cols) == 0:
            continue
        for j in cols:
            if i - last[j] >= H:
                last[j] = i
                entry = C[i + K, j]
                exit_ = C[i + K + H, j]
                if entry > 0 and exit_ > 0:
                    ret = (entry / exit_ - 1.0 - 2 * COST_SIDE) if UP else (exit_ / entry - 1.0 - 2 * COST_SIDE)
                    events.append((int(ts[i + K]), j, float(ret)))

    rets = np.array([e[2] for e in events])
    n = len(rets)
    up = rets[rets > 0]
    dn = rets[rets < 0]
    p_up = len(up) / n
    U = up.mean() * p_up
    D = -dn.mean() * (1 - p_up)
    label = f"插针做空（+{abs(PIN_TH):.0%} 急涨 / 持有 {HOLD_MIN}min / 成本 {COST_SIDE:.4f}）" if UP else \
            f"插针抄底（{PIN_TH:.0%} 插针 / 持有 {HOLD_MIN}min / 成本 {COST_SIDE:.4f}）"
    print(f"=== {label} ===")
    print(f"独立事件 {n:,}，均值 {rets.mean()*100:+.3f}% 中位 {np.median(rets)*100:+.3f}% "
          f"命中率 {p_up:.1%} 赔率比 {U/D:.2f}")

    print("--- 事件级分年 ---")
    yearly = {}
    for y in [2024, 2025, 2026]:
        sub = np.array([e[2] for e in events if pd.Timestamp(e[0], unit="s").year == y])
        if len(sub):
            up2 = sub[sub > 0]; dn2 = sub[sub < 0]
            U2 = up2.mean() * (len(up2)/len(sub)); D2 = -dn2.mean() * (len(dn2)/len(sub))
            print(f"  {y}: n={len(sub):>4} 均值{sub.mean()*100:+.2f}% 命中{(sub>0).mean():.0%} 赔率比{U2/D2:.2f}")
            yearly[y] = {"n": int(len(sub)), "mean": float(sub.mean()), "hit": float((sub>0).mean())}

    # 组合级：每事件固定仓位 POS_SIZE（诚实模型，非等权满仓复利）
    POS_SIZE = 0.05
    by_day: dict[int, list[float]] = {}
    for ts0, j, ret in events:
        by_day.setdefault(ts0 // 86400, []).append(ret)
    all_days = sorted(set(int(ts[t]) // 86400 for t in range(0, T, 1440)))
    dret = np.array([POS_SIZE * np.sum(by_day[d]) if d in by_day else 0.0 for d in all_days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    ann = eq[-1] ** (365 / len(dret)) - 1
    sharpe = dret.mean() / (dret.std() + 1e-12) * np.sqrt(365)
    active = sum(1 for d in all_days if d in by_day)
    print(f"\n--- 组合级（每事件 {POS_SIZE:.0%} 仓位，{len(dret)} 天，{active} 天有持仓，空仓 {(1-active/len(dret)):.0%}） ---")
    print(f"总收益 {eq[-1]-1:+.1%}  年化 {ann:+.1%}  Sharpe {sharpe:.2f}  最大回撤 {dd.min()*100:.1f}%")

    # 与慢动量 v2 的相关性
    corr = None
    try:
        import json as _json
        v2 = _json.load(open("data/model_out/replay_all.json", encoding="utf-8"))["legs"]["v2"]["equity"]
        v2_map = {int(x[0]) // 86400: float(x[1]) for x in v2}
        v2_days = sorted(v2_map)
        v2_ret = {v2_days[i+1]: v2_map[v2_days[i+1]] / v2_map[v2_days[i]] - 1
                  for i in range(len(v2_days) - 1)}
        common = sorted(set(by_day.keys()) & set(v2_ret.keys()))
        p = np.array([POS_SIZE * np.sum(by_day[d]) for d in common])
        v = np.array([v2_ret[d] for d in common])
        corr = float(np.corrcoef(p, v)[0, 1]) if len(common) > 5 else None
        print(f"与慢动量 v2 日收益相关 = {corr:+.3f}（{len(common)} 共同日）")
    except Exception as e:
        print(f"相关性计算失败: {e}")

    payload = {
        "params": {"pin_th": PIN_TH, "lookback": LOOKBACK, "hold_min": HOLD_MIN,
                   "cost_side": COST_SIDE, "pos_size": POS_SIZE},
        "events": int(n), "mean_ret": float(rets.mean()), "median_ret": float(np.median(rets)),
        "hit_rate": float(p_up), "odds": float(U / D),
        "total_ret": float(eq[-1] - 1), "annual": float(ann),
        "sharpe": float(sharpe), "max_dd": float(dd.min()),
        "corr_vs_v2": corr,
        "yearly": yearly,
        "equity": [[int(d), round(float(e), 4)] for d, e in zip(all_days, eq)],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘 {OUT}")


if __name__ == "__main__":
    main()
