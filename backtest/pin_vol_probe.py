# -*- coding: utf-8 -*-
"""插针 + 成交量提纯探测：放量插针(强平) vs 缩量插针(阴跌) 是否收益分化。

假说（坐实「赔率不对称=被迫清算」）：
  强平 = 清算引擎市价单砸出巨量 → 放量插针 → 超卖 → 强回归；
  阴跌 = 无买盘趋势下跌 → 缩量插针 → 更弱甚至延续。

方法：复用 pin_strategy 上升沿+去重找事件（只加载 close）；事件触发后逐个查 SQLite
拿该币过去15min量与前240min量，算 vol_ratio 分桶看 12h 收益分化。
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_ROWS = 1_000_000
PIN_TH = -0.05
LOOKBACK = 15
HOLD_MIN = 720
COST_SIDE = 0.0006
BASE_WIN = 240


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


def vol_query(conn: sqlite3.Connection, symbol: str, t0: int, t1: int) -> float:
    """返回 [t0, t1) 秒区间的成交量之和（复用连接）。"""
    v = conn.execute(
        "SELECT COALESCE(SUM(volume), 0) FROM klines_1m "
        "WHERE symbol=? AND open_time>=? AND open_time<?",
        (symbol, t0, t1)).fetchone()[0]
    return float(v)


def odds_ratio(r: np.ndarray) -> float:
    up = r[r > 0]; dn = r[r < 0]
    if len(up) == 0 or len(dn) == 0:
        return float("nan")
    return float(up.mean() * (len(up) / len(r)) / (-dn.mean() * (len(dn) / len(r))))


def main() -> None:
    C, ts, symbols = load_close()
    T, N = C.shape
    K = LOOKBACK
    H = HOLD_MIN
    W = BASE_WIN

    ret15 = C[K:] / C[:-K] - 1.0
    mask = ret15 <= PIN_TH
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] = mask[1:] & ~mask[:-1]
    edge[0] = mask[0]

    last = np.full(N, -10**9, dtype=np.int64)
    rows = []  # (ret, vol_ratio)
    conn = sqlite3.connect(DB)
    for i in range(T - K - H):
        cols = np.where(edge[i])[0]
        if len(cols) == 0:
            continue
        for j in cols:
            if i - last[j] >= H:
                last[j] = i
                t = i + K
                entry = C[t, j]; exit_ = C[t + H, j]
                if entry <= 0 or exit_ <= 0:
                    continue
                ret = exit_ / entry - 1.0 - 2 * COST_SIDE
                t_sec = int(ts[t])
                vr = np.nan
                if t_sec - W * 60 >= ts[0]:
                    vol15 = vol_query(conn, symbols[j], t_sec - K * 60, t_sec)
                    base = vol_query(conn, symbols[j], t_sec - W * 60, t_sec - K * 60) / (W / K)
                    vr = vol15 / base if base > 0 else 0.0
                rows.append((ret, vr, t_sec))
    conn.close()

    rets = np.array([r[0] for r in rows])
    vr = np.array([r[1] for r in rows], dtype=float)
    tss = np.array([r[2] for r in rows], dtype=np.int64)
    valid = ~np.isnan(vr)
    print(f"插针事件总数 {len(rets):,}（vol_ratio 有效 {valid.sum():,}）")
    print(f"全体: 均值 {rets.mean()*100:+.2f}% 命中 {(rets>0).mean():.1%} 赔率比 {odds_ratio(rets):.2f}")

    rv, vv = rets[valid], vr[valid]
    print(f"\n=== 按 vol_ratio（15min量 / 前4h平均15min量）分桶 ===")
    print(f"{'桶':>12} {'事件':>6} {'占比':>6} {'均值':>8} {'命中':>7} {'赔率比':>8}")
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (vv >= a) & (vv < b)
        if sel.sum() == 0:
            continue
        s = rv[sel]
        label = f"[{a},{b})" if b < 1e9 else f">={a}"
        print(f"{label:>12} {sel.sum():>6} {sel.sum()/len(rv):>6.0%} {s.mean()*100:>+8.2f}% "
              f"{(s>0).mean():>7.1%} {odds_ratio(s):>8.2f}")

    hi = rv[vv >= 2.0]
    lo = rv[vv < 1.0]
    print(f"\n=== 关键对比 ===")
    print(f"放量插针(vol>=2x): n={len(hi):>4} 均值{hi.mean()*100:+.2f}% 命中{(hi>0).mean():.1%} 赔率比{odds_ratio(hi):.2f}")
    print(f"缩量插针(vol<1x):  n={len(lo):>4} 均值{lo.mean()*100:+.2f}% 命中{(lo>0).mean():.1%} 赔率比{odds_ratio(lo):.2f}")

    # 分年验证甜点桶 [3.0,5.0) 是否跨年稳健
    print(f"\n=== 分年验证：甜点桶 [3.0,5.0) vs 全体 ===")
    sweet = (vv >= 3.0) & (vv < 5.0)
    year = np.array([pd.Timestamp(int(x), unit="s").year for x in tss])[valid]
    for y in [2024, 2025, 2026]:
        mask_y = year == y
        s_all = rv[mask_y]
        s_sw = rv[mask_y & sweet]
        if len(s_sw):
            print(f"  {y}: 全体 n={len(s_all):>4} 均值{s_all.mean()*100:+.2f}% 命中{(s_all>0).mean():.0%} | "
                  f"甜点[3,5) n={len(s_sw):>4} 均值{s_sw.mean()*100:+.2f}% 命中{(s_sw>0).mean():.0%} 赔率比{odds_ratio(s_sw):.2f}")
        else:
            print(f"  {y}: 甜点桶无事件")


if __name__ == "__main__":
    main()
