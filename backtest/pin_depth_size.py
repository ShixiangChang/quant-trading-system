# -*- coding: utf-8 -*-
"""插针深度加权下注：把「跌得越狠 → 赔率比越高」的剂量-反应关系转成仓位函数。

depth scan 已证明：深度 -3%→-15%，赔率比 1.32→15.69 单调递增，但事件数骤减、
Sharpe 反降（-5% 是实用甜点）。本实验回答：能不能把「深度」当连续信号，
对浅插针下小注、深插针下大注，在同样总资金预算下提升 Sharpe。

方法：基阈值 -3%（捕全部插针），每个事件记录深度 d=-ret15，按权重函数 w(d) 下注，
把 2 年总资金预算归一化到与「-5% 等权」基线相同（pos_scale = 5% × N(-5%) / Σw）。
Sharpe 对常数缩放不变，故相对权重决定风险调整收益。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_ROWS = 1_000_000
LOOKBACK = 15
HOLD_MIN = 720
COST_SIDE = 0.0006
POS_SIZE = 0.05
BASE_TH = -0.03
REF_TH = -0.05


def load_close() -> tuple[np.ndarray, np.ndarray, list[str]]:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol", values="close",
                           aggfunc="last").sort_index()[symbols]
    return close.values.astype(np.float64), close.index.values.astype(np.int64), symbols


def detect_events(C: np.ndarray, ts: np.ndarray, th: float) -> list[tuple[int, float, float]]:
    """返回 [(t_sec, depth, ret)]，上升沿触发 + HOLD_MIN 去重，depth = -ret15（正）。"""
    T, N = C.shape
    K = LOOKBACK
    H = HOLD_MIN
    ret15 = C[K:] / C[:-K] - 1.0
    mask = ret15 <= th
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] = mask[1:] & ~mask[:-1]
    edge[0] = mask[0]
    last = np.full(N, -10**9, dtype=np.int64)
    events: list[tuple[int, float, float]] = []
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
                    depth = float(-ret15[i, j])  # 正值，越大越深
                    ret = float(exit_ / entry - 1.0 - 2 * COST_SIDE)
                    events.append((int(ts[i + K]), depth, ret))
    return events


def odds_ratio(r: np.ndarray) -> float:
    up = r[r > 0]; dn = r[r < 0]
    if len(up) == 0 or len(dn) == 0:
        return float("nan")
    return float(up.mean() * (len(up) / len(r)) / (-dn.mean() * (len(dn) / len(r))))


def evaluate(events: list[tuple[int, float, float]], wfn, pos_scale: float,
             all_days: list[int]) -> dict:
    by_day: dict[int, float] = {}
    for t0, d, ret in events:
        by_day[t0 // 86400] = by_day.get(t0 // 86400, 0.0) + wfn(d) * ret
    dret = np.array([pos_scale * by_day[d] if d in by_day else 0.0 for d in all_days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    sharpe = float(dret.mean() / (dret.std() + 1e-12) * np.sqrt(365))
    ann = float(eq[-1] ** (365 / len(dret)) - 1)
    return {"total": float(eq[-1] - 1), "annual": ann, "sharpe": sharpe, "dd": dd,
            "gross": float(sum(wfn(d) for _, d, _ in events))}


def main() -> None:
    C, ts, symbols = load_close()
    T = C.shape[0]
    all_days = sorted(set(int(ts[t]) // 86400 for t in range(0, T, 1440)))
    print(f"{len(symbols)} 币全量 / {T:,} 分钟 / 持有 {HOLD_MIN//60}h / 成本 {COST_SIDE:.4f}\n")

    # 1) 等权阈值扫描（复现 depth scan，验证基线）
    print("=== 等权阈值扫描（复现 depth scan，验证）===")
    print(f"{'阈值':>7} {'事件':>6} {'均值':>8} {'命中':>6} {'赔率比':>7} {'Sharpe':>7} {'回撤':>8}")
    base_events: dict[float, list] = {}
    for th in [-0.03, -0.04, -0.05, -0.06, -0.08, -0.10, -0.12, -0.15]:
        ev = detect_events(C, ts, th)
        base_events[th] = ev
        rets = np.array([r for _, _, r in ev])
        if len(rets) == 0:
            continue
        r = evaluate(ev, lambda d: 1.0, POS_SIZE, all_days)
        print(f"{th:>7.0%} {len(rets):>6} {rets.mean()*100:>+8.2f}% {(rets>0).mean():>6.1%} "
              f"{odds_ratio(rets):>7.2f} {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}%")

    # 2) 深度加权下注（基阈值 -3%，总资金归一化到 -5% 等权基线）
    ref = base_events[REF_TH]
    ref_gross = sum(1.0 for _ in ref)          # = N(-5%)
    ev3 = base_events[BASE_TH]
    n3 = len(ev3)
    print(f"\n=== 深度加权下注（基阈值 {BASE_TH:.0%}，{n3} 事件，总资金归一化到 {len(ref)} 笔 × 5%）===")

    weights = {
        "等权(基线)": lambda d: 1.0,
        "线性 w=d/5%": lambda d: np.clip(d / 0.05, 0.0, 3.0),
        "平方根 w=√(d/5%)": lambda d: np.clip(np.sqrt(d / 0.05), 0.0, 3.0),
        "对数 w=1+ln(d/3%)": lambda d: np.clip(1.0 + np.log(d / 0.03), 0.0, 3.0),
    }
    print(f"{'方案':>22} {'总收益':>9} {'年化':>9} {'Sharpe':>7} {'回撤':>8} {'深插针占比(≥8%)':>16}")
    for name, wfn in weights.items():
        gross = sum(wfn(d) for _, d, _ in ev3)
        pos_scale = POS_SIZE * ref_gross / gross
        r = evaluate(ev3, wfn, pos_scale, all_days)
        deep_cap = sum(wfn(d) for _, d, _ in ev3 if d >= 0.08)
        print(f"{name:>22} {r['total']*100:>+9.1f}% {r['annual']*100:>+9.1f}% "
              f"{r['sharpe']:>7.2f} {r['dd']*100:>8.1f}% {deep_cap/gross:>15.1%}")

    # 3) 参考：-5% 等权基线（当前策略）
    r_ref = evaluate(ref, lambda d: 1.0, POS_SIZE, all_days)
    print(f"\n=== 当前策略（-5% 等权，{len(ref)} 事件 × 5%）===")
    print(f"总收益 {r_ref['total']*100:+.1f}%  年化 {r_ref['annual']*100:+.1f}%  "
          f"Sharpe {r_ref['sharpe']:.2f}  回撤 {r_ref['dd']*100:.1f}%")


if __name__ == "__main__":
    main()
