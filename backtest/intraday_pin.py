# -*- coding: utf-8 -*-
"""插针赔率不对称实验：急跌（对手方被迫成交/爆仓清算）后的收益分布。

本实验假设：收益来源 = 赔率不对称（事件驱动的对手方被迫成交），而非预测准确。
本实验回答一个具体问题：急跌 X% 之后，未来 H 分钟的收益分布是右偏（期望上行 > 期望下行，
适合小仓抄底）还是左偏（期望下行 > 期望上行，左侧抄底风险高）？

指标（赔率框架）：
  U = E[fwd | fwd>0] * P(fwd>0)   期望上行
  D = -E[fwd | fwd<0] * P(fwd<0)  期望下行（损失）
  赔率比 = U / D。>1 = 期望上行占优；<1 = 期望下行占优。

无前视：信号用 t 之前 15min，收益用 t 之后。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_ROWS = 1_000_000   # 只用完整 2 年的币


def load_close() -> tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol", values="close",
                           aggfunc="last").sort_index()[symbols]
    C = close.values.astype(float)
    ts = close.index.values.astype(np.int64)
    return C, ts


def odds_ratio(fwd: np.ndarray) -> dict:
    """给定未来收益数组，返回赔率结构。"""
    f = fwd[np.isfinite(fwd)]
    if len(f) < 50:
        return {"n": len(f)}
    up = f[f > 0]
    dn = f[f < 0]
    p_up = len(up) / len(f)
    U = up.mean() * p_up if len(up) else 0.0
    D = -dn.mean() * (1 - p_up) if len(dn) else 0.0
    return {
        "n": len(f),
        "mean": float(f.mean()),
        "median": float(np.median(f)),
        "skew": float(pd.Series(f).skew()),
        "p_up": float(p_up),
        "U": float(U),
        "D": float(D),
        "odds": float(U / D) if D > 0 else float("inf"),
        "p5": float(np.percentile(f, 5)),
        "p95": float(np.percentile(f, 95)),
    }


def main() -> None:
    C, ts = load_close()
    T, N = C.shape
    print(f"{N} 币完整 2 年，{T:,} 分钟截面")

    # 过去 15min 收益（信号）
    K = 15
    ret15 = C[K:] / C[:-K] - 1.0          # shape (T-K, N)

    PIN_THRESH = [-0.03, -0.05, -0.08]
    HORIZONS = [60, 240, 1440]            # 1h / 4h / 24h

    for th in PIN_THRESH:
        # 插针掩码：过去 15min 跌超 th
        mask = ret15 <= th
        n_pins = int(mask.sum())
        if n_pins == 0:
            continue
        print(f"\n=== 插针：过去15min跌超 {th:+.0%}（共 {n_pins:,} 次） ===")
        for H in HORIZONS:
            # 未来 H 分钟收益：fwd[t] 对应 t 时刻（ret15 的 t 是 C[t+K] 时刻）
            # ret15[i] 用的是 C[i..i+K]，所以 t = i+K。未来 H = C[i+K+H]/C[i+K]-1
            # 构造 fwd：对每个 i，未来 H 收益
            fwd = np.full((T - K, N), np.nan)
            fwd[: T - K - H] = C[K + H:] / C[K: T - H] - 1.0
            vals = fwd[mask]
            r = odds_ratio(vals)
            if "mean" not in r:
                print(f"  未来{H // 60}h: 样本不足")
                continue
            label = f"{H // 60}h" if H < 1440 else "24h"
            print(f"  未来{label:>3}: n={r['n']:>7,} 均值{r['mean']*100:+.3f}% "
                  f"中位{r['median']*100:+.3f}% 偏度{r['skew']:+.2f} "
                  f"P涨{r['p_up']:.0%} 赔率比{r['odds']:.2f} "
                  f"P5={r['p5']*100:+.1f}% P95={r['p95']*100:+.1f}%")


if __name__ == "__main__":
    main()
