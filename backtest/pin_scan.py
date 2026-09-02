# -*- coding: utf-8 -*-
"""插针抄底策略参数稳健性扫描。

扫描网格：
  - 插针阈值 pin_th: -3% / -4% / -5% / -6% / -8%
  - 持有期 hold_min: 60 / 120 / 240 / 480 / 720 / 1440 分钟
  - 成本 cost_side: 3bp / 6bp / 10bp（单边）

对每个组合，用与 pin_strategy.py 完全一致的「上升沿触发 + 4h 去重 + 每事件 5% 仓位」
算独立事件数 / 均值 / 命中率 / 赔率比 / 组合年化 / Sharpe / 最大回撤。

目的：确认 edge 是否只在某个参数点上成立（过拟合），还是在一个「高原」上稳健。
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_ROWS = 1_000_000
LOOKBACK = 15           # 过去 15min 收益（固定，来自 pin_strategy 的发现）
POS_SIZE = 0.05
OUT = "data/model_out/pin_scan.json"

PIN_THS = [-0.03, -0.04, -0.05, -0.06, -0.08]
HOLD_MINS = [60, 120, 240, 480, 720, 1440]
COST_SIDES = [0.0003, 0.0006, 0.0010]


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


def portfolio_metrics(days: list[int], events_by_day: dict[int, list[float]],
                      pos: float) -> dict:
    """组合级诚实模型：每事件固定仓位 pos，日收益 = pos * sum(当日事件收益)。"""
    dret = np.array([pos * np.sum(events_by_day[d]) if d in events_by_day else 0.0
                     for d in days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    ann = eq[-1] ** (365 / max(len(dret), 1)) - 1
    sharpe = dret.mean() / (dret.std() + 1e-12) * np.sqrt(365)
    return {"total": float(eq[-1] - 1), "annual": float(ann),
            "sharpe": float(sharpe), "max_dd": float(dd.min())}


def main() -> None:
    C, ts, symbols = load_close()
    T, N = C.shape
    K = LOOKBACK
    ret15 = C[K:] / C[:-K] - 1.0          # (T-K, N)
    ts_trim = ts[K:]

    all_days = sorted(set(int(t) // 86400 for t in ts_trim))

    # 预收集每个 (pin_th, H) 的事件（用原始收益，成本事后减）
    results: list[dict] = []
    for pin_th in PIN_THS:
        mask = ret15 <= pin_th
        edge = np.zeros_like(mask, dtype=bool)
        edge[1:] = mask[1:] & ~mask[:-1]
        edge[0] = mask[0]
        for H in HOLD_MINS:
            last = np.full(N, -10**9, dtype=np.int64)
            raw = []                       # (day_idx, raw_ret)
            per_symbol_pos = np.zeros(N, dtype=int)
            per_symbol_ret = np.zeros(N, dtype=float)
            per_symbol_n = np.zeros(N, dtype=int)
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
                            r = exit_ / entry - 1.0
                            day = int(ts_trim[i + K]) // 86400
                            raw.append((day, r))
                            per_symbol_ret[j] += r
                            per_symbol_n[j] += 1
                            per_symbol_pos[j] += (r > 0)
            for cost in COST_SIDES:
                by_day: dict[int, list[float]] = {}
                for day, r in raw:
                    by_day.setdefault(day, []).append(r - 2 * cost)
                rets = np.array([r - 2 * cost for _, r in raw]) if raw else np.array([])
                n = len(rets)
                if n == 0:
                    continue
                up = rets[rets > 0]
                dn = rets[rets < 0]
                p_up = len(up) / n
                U = up.mean() * p_up if len(up) else 0.0
                D = -dn.mean() * (1 - p_up) if len(dn) else 0.0
                odds = U / D if D > 0 else float("inf")
                pm = portfolio_metrics(all_days, by_day, POS_SIZE)
                # 分币一致性：有多少币正收益
                nsym_pos = int(np.sum((per_symbol_n > 0) & (per_symbol_ret - 2 * cost * per_symbol_n > 0)))
                nsym_active = int(np.sum(per_symbol_n > 0))
                results.append({
                    "pin_th": pin_th, "hold_min": H, "cost_side": cost,
                    "n": n, "mean": float(rets.mean()), "median": float(np.median(rets)),
                    "hit": float(p_up), "odds": float(odds),
                    "sym_pos": nsym_pos, "sym_active": nsym_active,
                    **pm,
                })

    # 打印表格（按 Sharpe 降序，展示高原）
    df = pd.DataFrame(results)
    print(f"共 {len(df)} 个组合（{len(PIN_THS)} 阈值 × {len(HOLD_MINS)} 持有期 × {len(COST_SIDES)} 成本）\n")
    cols = ["pin_th", "hold_min", "cost_side", "n", "mean", "hit", "odds",
            "sym_pos", "annual", "sharpe", "max_dd"]
    show = df[cols].copy()
    show["mean"] = (show["mean"] * 100).round(2)
    show["hit"] = (show["hit"] * 100).round(0)
    show["odds"] = show["odds"].round(2)
    show["annual"] = (show["annual"] * 100).round(1)
    show["sharpe"] = show["sharpe"].round(2)
    show["max_dd"] = (show["max_dd"] * 100).round(1)
    show = show.sort_values("sharpe", ascending=False)
    print(show.to_string(index=False))

    # 默认成本 6bp 下的稳健性总结
    print("\n=== 成本 6bp 下，各阈值×持有期 的 Sharpe 矩阵 ===")
    pivot = df[df.cost_side == 0.0006].pivot_table(
        index="pin_th", columns="hold_min", values="sharpe", aggfunc="first")
    print(pivot.round(2).to_string())

    print("\n=== 成本 6bp 下，各阈值×持有期 的年化收益(%) 矩阵 ===")
    pivot2 = df[df.cost_side == 0.0006].pivot_table(
        index="pin_th", columns="hold_min", values="annual", aggfunc="first")
    print((pivot2 * 100).round(1).to_string())

    # 事件数矩阵
    print("\n=== 成本 6bp 下，各阈值×持有期 的独立事件数 ===")
    pivot3 = df[df.cost_side == 0.0006].pivot_table(
        index="pin_th", columns="hold_min", values="n", aggfunc="first")
    print(pivot3.astype(int).to_string())

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"\n已落盘 {OUT}")


if __name__ == "__main__":
    main()
