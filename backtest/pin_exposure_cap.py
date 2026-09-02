# -*- coding: utf-8 -*-
"""插针仓位上限：对比「实时可执行」的并发上限 vs「回看」的每日保留最深。

核心发现：-5%/15min 插针 816 事件，最多一天 34 币同时插针（=170% 隐式杠杆）。
回测若按「每日保留最深 20 笔」上限，Sharpe 2.16→2.55、回撤 -11.9%→-6.5%（但用了当日回看，
实时做不到）。本脚本补两个实时可执行的版本：

1. 并发上限（FCFS）：同时最多持有 N 笔，满了就跳过新插针（实时可实现）。
2. 每日上限（FCFS）：单日最多开 N 笔，满了跳过（实时可实现）。

对比三者的 Sharpe/回撤/收益，确认「实时版本」是否同样改善。
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


def daily_ret_from_used(used, all_days):
    """used: list of (t_sec, ret) 实际开仓的。"""
    by_day = {}
    for t0, r in used:
        by_day[t0 // 86400] = by_day.get(t0 // 86400, 0.0) + POS_SIZE * r
    dret = np.array([by_day[d] if d in by_day else 0.0 for d in all_days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    return {
        "sharpe": float(dret.mean() / (dret.std() + 1e-12) * np.sqrt(365)),
        "dd": float(((eq - peak) / peak).min()),
        "total": float(eq[-1] - 1),
        "annual": float(eq[-1] ** (365 / len(dret)) - 1),
    }


def eval_concurrent(events, all_days, cap):
    """并发上限 FCFS：同时最多 cap 笔（12h 持有）。实时可实现。"""
    events = sorted(events, key=lambda x: x[0])
    open_pos = []  # (close_t, ret)
    used = []
    for t0, d, r in events:
        close_t = t0 + HOLD_MIN * 60
        open_pos = [p for p in open_pos if p[0] > t0]
        if len(open_pos) < cap:
            open_pos.append((close_t, r))
            used.append((t0, r))
    return daily_ret_from_used(used, all_days), len(used)


def eval_daily_fcfs(events, all_days, cap):
    """每日上限 FCFS：单日最多 cap 笔，满了跳过。实时可实现。"""
    events = sorted(events, key=lambda x: x[0])
    day_count = {}
    used = []
    for t0, d, r in events:
        dd = t0 // 86400
        day_count[dd] = day_count.get(dd, 0) + 1
        if day_count[dd] <= cap:
            used.append((t0, r))
    return daily_ret_from_used(used, all_days), len(used)


def eval_daily_deepest(events, all_days, cap):
    """每日保留最深 cap 笔（回看，实时做不到，仅作上界参考）。"""
    day_events = {}
    for t0, d, r in events:
        day_events.setdefault(t0 // 86400, []).append((d, r))
    used = []
    for day, evs in day_events.items():
        for d, r in sorted(evs, key=lambda x: -x[0])[:cap]:
            used.append((day * 86400, r))
    return daily_ret_from_used(used, all_days), sum(min(len(v), cap) for v in day_events.values())


def main():
    C, ts, symbols = load_close()
    T = C.shape[0]
    all_days = sorted(set(int(ts[t]) // 86400 for t in range(0, T, 1440)))
    ret = C[LOOKBACK:] / C[:-LOOKBACK] - 1.0
    events = detect(C, ts, ret, TH, LOOKBACK, HOLD_MIN)
    print(f"{len(symbols)} 币 / {TH:.0%} 插针 / 持有 {HOLD_MIN//60}h / 事件 {len(events)} 笔\n")

    print(f"{'上限方案':>22} {'用笔数':>6} {'总收益':>9} {'年化':>9} {'Sharpe':>7} {'回撤':>8}")
    # 无上限基线
    r, n = eval_concurrent(events, all_days, cap=10**9)
    print(f"{'无上限':>22} {n:>6} {r['total']*100:>+9.1f}% {r['annual']*100:>+9.1f}% {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}%")

    for cap in [20, 10, 5]:
        rc, nc = eval_concurrent(events, all_days, cap)
        rf, nf = eval_daily_fcfs(events, all_days, cap)
        rd, nd = eval_daily_deepest(events, all_days, cap)
        print(f"{'并发上限'+str(cap)+'笔(FCFS)':>22} {nc:>6} {rc['total']*100:>+9.1f}% {rc['annual']*100:>+9.1f}% {rc['sharpe']:>7.2f} {rc['dd']*100:>7.1f}%")
        print(f"{'每日上限'+str(cap)+'笔(FCFS)':>22} {nf:>6} {rf['total']*100:>+9.1f}% {rf['annual']*100:>+9.1f}% {rf['sharpe']:>7.2f} {rf['dd']*100:>7.1f}%")
        print(f"{'每日保留最深'+str(cap)+'笔(回看)':>22} {nd:>6} {rd['total']*100:>+9.1f}% {rd['annual']*100:>+9.1f}% {rd['sharpe']:>7.2f} {rd['dd']*100:>7.1f}%")


if __name__ == "__main__":
    main()
