# -*- coding: utf-8 -*-
"""插针深度谱系扫描：验证「跌得越狠 → 回归越猛（清算越彻底）」的剂量-反应关系。

用 33 币全量 1m 数据，扫插针阈值 -3% ~ -15%，固定持有 12h、每事件 5% 仓位、
上升沿触发 + 12h 去重（与 pin_strategy 一致）。若深度→赔率比单调递增，则坐实
edge 机制 =「清算越彻底、超卖越深、回归越强」，而非巧合。
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
THRESHOLDS = [-0.03, -0.04, -0.05, -0.06, -0.08, -0.10, -0.12, -0.15]


def load_close() -> tuple[np.ndarray, np.ndarray, list[str]]:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol", values="close",
                           aggfunc="last").sort_index()[symbols]
    ts = close.index.values.astype(np.int64)
    return close.values.astype(np.float64), ts, symbols


def odds_ratio(r: np.ndarray) -> float:
    up = r[r > 0]; dn = r[r < 0]
    if len(up) == 0 or len(dn) == 0:
        return float("nan")
    return float(up.mean() * (len(up) / len(r)) / (-dn.mean() * (len(dn) / len(r))))


def run_threshold(C: np.ndarray, ts: np.ndarray, th: float) -> dict:
    T, N = C.shape
    K = LOOKBACK
    H = HOLD_MIN
    ret15 = C[K:] / C[:-K] - 1.0
    mask = ret15 <= th
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] = mask[1:] & ~mask[:-1]
    edge[0] = mask[0]

    last = np.full(N, -10**9, dtype=np.int64)
    events = []
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
                    events.append((int(ts[i + K]), float(exit_ / entry - 1.0 - 2 * COST_SIDE)))

    rets = np.array([e[1] for e in events])
    if len(rets) == 0:
        return {"th": th, "n": 0}

    # 组合级（每事件 5% 仓位，按日归组）
    by_day: dict[int, list[float]] = {}
    for t0, ret in events:
        by_day.setdefault(t0 // 86400, []).append(ret)
    all_days = sorted(set(int(ts[t]) // 86400 for t in range(0, T, 1440)))
    dret = np.array([POS_SIZE * np.sum(by_day[d]) if d in by_day else 0.0 for d in all_days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    sharpe = float(dret.mean() / (dret.std() + 1e-12) * np.sqrt(365))

    yearly = {}
    for y in [2024, 2025, 2026]:
        sub = np.array([e[1] for e in events if pd.Timestamp(e[0], unit="s").year == y])
        if len(sub):
            yearly[y] = (len(sub), sub.mean(), (sub > 0).mean())

    return {
        "th": th, "n": len(rets), "mean": float(rets.mean()),
        "hit": float((rets > 0).mean()), "odds": odds_ratio(rets),
        "sharpe": sharpe, "dd": dd, "total": float(eq[-1] - 1),
        "yearly": yearly,
    }


def main() -> None:
    C, ts, symbols = load_close()
    print(f"33 币全量 / {len(ts)} 分钟 / 持有 {HOLD_MIN//60}h / 每事件 {POS_SIZE:.0%} 仓位\n")
    print(f"{'阈值':>7} {'事件':>6} {'均值':>8} {'命中':>6} {'赔率比':>7} {'总收益':>8} {'Sharpe':>7} {'回撤':>8}")
    for th in THRESHOLDS:
        r = run_threshold(C, ts, th)
        if r["n"] == 0:
            print(f"{th:>7.0%} {'无事件':>6}")
            continue
        print(f"{th:>7.0%} {r['n']:>6} {r['mean']*100:>+8.2f}% {r['hit']:>6.1%} {r['odds']:>7.2f} "
              f"{r['total']*100:>+8.1f}% {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}%")

    # 深度 vs 赔率的单调性
    print("\n=== 深度 → 赔率比（看是否单调递增）===")
    for th in THRESHOLDS:
        r = run_threshold(C, ts, th)
        if r["n"]:
            y = r["yearly"]
            ystr = "  ".join(f"{yy}:{y[yy][0] if yy in y else 0}个/{y[yy][1]*100:+.1f}%" if yy in y else f"{yy}:-" for yy in [2024, 2025, 2026])
            print(f"  {th:>7.0%}  赔率比 {r['odds']:.2f}  均值{r['mean']*100:+.2f}%  n={r['n']}  |  {ystr}")


if __name__ == "__main__":
    main()
