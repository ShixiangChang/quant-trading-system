# -*- coding: utf-8 -*-
"""插针深度加权（-5% 基阈值，实时可执行）——收口测试。

前面已排除：
- 深度加权@-3% 基阈值（含浅插针）：Sharpe 1.41，远不如 -5% 过滤 2.16。
- 仓位硬上限 FCFS（实时）：Sharpe 1.90/2.06，比无上限 2.16 差。
- 每日保留最深（回看作弊）：2.55，但实时做不到。

本测试最后补一个实时可执行变体：-5% 基阈值（不含浅插针），对深插针连续加权，
总资金归一化到与「等权」相同。若仍不超 2.16，则结论钉死：-5% 等权就是实时最优。
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
TH = -0.05


def load_close():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol", values="close",
                           aggfunc="last").sort_index()[symbols]
    return close.values.astype(np.float64), close.index.values.astype(np.int64), symbols


def detect(C, ts, ret, th, K, H):
    T, N = C.shape
    mask = ret <= th
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] = mask[1:] & ~mask[:-1]
    edge[0] = mask[0]
    rows, cols = np.where(edge)
    last = np.full(N, -10**9, dtype=np.int64)
    events = []
    for i, j in zip(rows, cols):
        if i >= T - K - H:
            continue
        if i - last[j] >= H:
            last[j] = i
            entry = C[i + K, j]; exit_ = C[i + K + H, j]
            if entry > 0 and exit_ > 0:
                events.append((int(ts[i + K]), float(-ret[i, j]), float(exit_ / entry - 1.0 - 2 * COST_SIDE)))
    return events


def evaluate(events, all_days, wfn):
    by_day = {}
    gross = sum(wfn(d) for _, d, _ in events)
    pos_scale = POS_SIZE * len(events) / gross  # 总资金归一化到等权
    for t0, d, r in events:
        by_day[t0 // 86400] = by_day.get(t0 // 86400, 0.0) + pos_scale * wfn(d) * r
    dret = np.array([by_day[d] if d in by_day else 0.0 for d in all_days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    return {
        "sharpe": float(dret.mean() / (dret.std() + 1e-12) * np.sqrt(365)),
        "dd": float(((eq - peak) / peak).min()),
        "total": float(eq[-1] - 1),
        "annual": float(eq[-1] ** (365 / len(dret)) - 1),
    }


def main():
    C, ts, symbols = load_close()
    T = C.shape[0]
    all_days = sorted(set(int(ts[t]) // 86400 for t in range(0, T, 1440)))
    ret = C[LOOKBACK:] / C[:-LOOKBACK] - 1.0
    events = detect(C, ts, ret, TH, LOOKBACK, HOLD_MIN)
    print(f"{len(symbols)} 币 / {TH:.0%} 插针 / 持有 {HOLD_MIN//60}h / 事件 {len(events)} 笔 / 成本 {COST_SIDE:.4f}\n")

    weights = {
        "等权(基线)": lambda d: 1.0,
        "线性 w=d/5%": lambda d: np.clip(d / 0.05, 0.0, 3.0),
        "平方根 w=√(d/5%)": lambda d: np.clip(np.sqrt(d / 0.05), 0.0, 3.0),
        "平方 w=(d/5%)²": lambda d: np.clip((d / 0.05) ** 2, 0.0, 6.0),
    }
    print(f"{'方案':>22} {'总收益':>9} {'年化':>9} {'Sharpe':>7} {'回撤':>8}")
    for name, wfn in weights.items():
        r = evaluate(events, all_days, wfn)
        print(f"{name:>22} {r['total']*100:>+9.1f}% {r['annual']*100:>+9.1f}% {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}%")


if __name__ == "__main__":
    main()
