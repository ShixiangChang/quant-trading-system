# -*- coding: utf-8 -*-
"""插针回看窗口扫描：之前只扫了阈值×持有×成本，回看一直固定 15min。

「急跌」的定义（回看 K 分钟跌超阈值）直接影响捕捉哪种被迫成交：
- 5min/10min：尖锐 flash crash（闪电强平）
- 15min：标准插针（当前）
- 30min/60min：持续瀑布（级联清算）

用 33 币全量 1m 数据，扫 回看×阈值（持有固定 12h 已证最优），找更优的「急跌」定义。
同时输出 -5%/15min 基线的聚簇风险（同日最多同时插针数/当日总仓位）。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_ROWS = 1_000_000
HOLD_MIN = 720
COST_SIDE = 0.0006
POS_SIZE = 0.05
LOOKBACKS = [5, 10, 15, 30, 60]
THRESHOLDS = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.08]


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


def detect(C: np.ndarray, ts: np.ndarray, ret: np.ndarray, th: float, K: int, H: int):
    """向量化事件检测（上升沿触发 + 每币 H 分钟去重），返回 [(t_sec, depth, ret)]。"""
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
            entry = C[i + K, j]
            exit_ = C[i + K + H, j]
            if entry > 0 and exit_ > 0:
                events.append((int(ts[i + K]), float(-ret[i, j]), float(exit_ / entry - 1.0 - 2 * COST_SIDE)))
    return events


def odds_ratio(r: np.ndarray) -> float:
    up = r[r > 0]; dn = r[r < 0]
    if len(up) == 0 or len(dn) == 0:
        return float("nan")
    return float(up.mean() * (len(up) / len(r)) / (-dn.mean() * (len(dn) / len(r))))


def evaluate(events, all_days):
    if not events:
        return None
    by_day = {}
    for t0, d, r in events:
        by_day[t0 // 86400] = by_day.get(t0 // 86400, 0.0) + POS_SIZE * r
    dret = np.array([by_day[d] if d in by_day else 0.0 for d in all_days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    sharpe = float(dret.mean() / (dret.std() + 1e-12) * np.sqrt(365))
    rets = np.array([r for _, _, r in events])
    return {"n": len(events), "mean": float(rets.mean()), "hit": float((rets > 0).mean()),
            "odds": odds_ratio(rets), "sharpe": sharpe, "dd": dd,
            "total": float(eq[-1] - 1), "annual": float(eq[-1] ** (365 / len(dret)) - 1)}


def main():
    C, ts, symbols = load_close()
    T = C.shape[0]
    all_days = sorted(set(int(ts[t]) // 86400 for t in range(0, T, 1440)))
    print(f"{len(symbols)} 币 / {T:,} 分钟 / 持有 {HOLD_MIN//60}h / 成本 {COST_SIDE:.4f}\n")

    print("=== 回看 × 阈值 扫描（Sharpe，持有 12h 固定）===")
    print(f"{'回看':>5} | " + " ".join(f"{th:>8.0%}" for th in THRESHOLDS))
    results = {}
    for K in LOOKBACKS:
        ret = C[K:] / C[:-K] - 1.0
        row = []
        for th in THRESHOLDS:
            ev = detect(C, ts, ret, th, K, HOLD_MIN)
            r = evaluate(ev, all_days)
            results[(K, th)] = (ev, r)
            row.append(f"{r['sharpe']:>8.2f}" if r and r['n'] >= 5 else "     --")
        print(f"{K:>4}m | " + " ".join(row))

    print("\n=== 各回看最优阈值（按 Sharpe，事件数≥50）===")
    print(f"{'回看':>5} {'阈值':>7} {'事件':>6} {'均值':>8} {'命中':>6} {'赔率比':>7} {'Sharpe':>7} {'回撤':>8} {'年化':>8}")
    for K in LOOKBACKS:
        best = None
        for th in THRESHOLDS:
            ev, r = results[(K, th)]
            if r and r["n"] >= 50 and (best is None or r["sharpe"] > best[0]["sharpe"]):
                best = (r, th, ev)
        if best:
            r, th, ev = best
            print(f"{K:>4}m {th:>7.0%} {r['n']:>6} {r['mean']*100:>+8.2f}% {r['hit']:>6.1%} "
                  f"{r['odds']:>7.2f} {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}% {r['annual']*100:>+7.1f}%")

    # 聚簇风险（-5%/15min 基线）
    print("\n=== 聚簇风险（-5% / 15min 基线）===")
    ev, r = results[(15, -0.05)]
    day_events = {}
    for t0, d, _ in ev:
        day_events[t0 // 86400] = day_events.get(t0 // 86400, 0) + 1
    cnts = np.array(list(day_events.values()))
    print(f"事件总数 {len(ev)}，分布在 {len(day_events)} 天")
    print(f"日均 {cnts.mean():.2f} 笔，最多一天 {cnts.max()} 笔（= 单日 {cnts.max()*POS_SIZE:.0%} 仓位）")
    print(f"单日 ≥3 笔的天数: {(cnts >= 3).sum()} 天，≥5 笔: {(cnts >= 5).sum()} 天，≥8 笔: {(cnts >= 8).sum()} 天")
    top_days = sorted(day_events.items(), key=lambda x: -x[1])[:5]
    for d, c in top_days:
        print(f"  {pd.Timestamp(d*86400, unit='s').date()}  {c} 笔同时插针")


if __name__ == "__main__":
    main()
