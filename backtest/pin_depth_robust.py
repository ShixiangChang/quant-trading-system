# -*- coding: utf-8 -*-
"""深度加权稳健性检验：分年验证 + 指数谱系。

上轮发现 -5% 基阈值上深度加权（平方 w=(d/5%)²）Sharpe 2.16→2.37、回撤 -11.9%→-7.5%。
但这是样本内选优（4 个权函数里挑的）。本脚本做两件事：
1. 分年验证：平方加权是年年都改善，还是某一年独有？
2. 指数谱系 0→0.5→1→2→3：Sharpe 是继续单调升（危险，越极端越好=过拟合），
   还是在 2 附近见顶回落（甜点，可信）？
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


def make_wfn(exp):
    if exp == 0:
        return lambda d: 1.0
    return lambda d: np.clip((d / 0.05) ** exp, 0.0, 10.0)


def evaluate(events, all_days, wfn):
    by_day = {}
    gross = sum(wfn(d) for _, d, _ in events)
    pos_scale = POS_SIZE * len(events) / max(gross, 1e-12)
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
    print(f"{len(symbols)} 币 / {TH:.0%} 插针 / 事件 {len(events)} 笔\n")

    # 1) 指数谱系
    print("=== 指数谱系（0=等权，越大越集中到深插针）===")
    print(f"{'指数':>5} {'Sharpe':>7} {'回撤':>8} {'总收益':>9} {'年化':>8}")
    for exp in [0, 0.5, 1, 2, 3]:
        r = evaluate(events, all_days, make_wfn(exp))
        print(f"{exp:>5} {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}% {r['total']*100:>+9.1f}% {r['annual']*100:>+8.1f}%")

    # 2) 分年验证（等权 vs 平方）
    print("\n=== 分年验证（等权 vs 平方 w=(d/5%)²）===")
    for name, wfn in [("等权", make_wfn(0)), ("平方", make_wfn(2))]:
        print(f"  {name}:")
        for y in [2024, 2025, 2026]:
            sub = [e for e in events if pd.Timestamp(e[0], unit="s").year == y]
            if not sub:
                continue
            rs = np.array([r for _, _, r in sub])
            # 加权平均收益（用权函数）
            w = np.array([wfn(d) for _, d, _ in sub])
            wmean = float(np.average(rs, weights=w))
            wht = float(np.average((rs > 0).astype(float), weights=w))
            up = rs[rs > 0]; dn = rs[rs < 0]
            odds = float(up.mean() * (len(up)/len(rs)) / (-dn.mean() * (len(dn)/len(rs)))) if len(dn) else float('nan')
            print(f"    {y}: n={len(sub):>4} 加权均值{wmean*100:+.2f}% 加权命中{wht:.0%} 等权均值{rs.mean()*100:+.2f}% 赔率比{odds:.2f}")

    # 3) 深度分布（平方权重的资金集中度）
    print("\n=== 平方权重的资金集中度 ===")
    wfn = make_wfn(2)
    depths = np.array([d for _, d, _ in events])
    weights = np.array([wfn(d) for _, d, _ in events])
    total = weights.sum()
    for lo, hi in [(0.05, 0.06), (0.06, 0.08), (0.08, 0.10), (0.10, 0.15), (0.15, 1.0)]:
        m = (depths >= lo) & (depths < hi)
        print(f"  深度 {lo:.0%}~{hi:.0%}: {m.sum():>4} 事件, 资金占比 {weights[m].sum()/total:.1%}")


if __name__ == "__main__":
    main()
