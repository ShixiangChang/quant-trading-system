"""
日内反转多空回测(严谨版)—— 延迟1分钟成交消除 bid-ask bounce,多持有期扫描。

信号: t 时刻 close 的过去 L 分钟收益排名。
成交: t+1 分钟 close(延迟 1 分钟,消除 bounce + look-ahead)。
兑现: t+1+HOLD 分钟 close。
"""
import sqlite3
import numpy as np
import pandas as pd
import datetime

DB = "data/monitor.db"
MIN_ROWS = 900_000


def main():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol",
                           values="close", aggfunc="last").sort_index()
    close = close[symbols]
    ts = close.index.values.astype(np.int64)
    C = close.values.astype(float)
    T, N = C.shape
    print(f"close 矩阵 {T:,} 分钟 × {N} 币")
    print(f"{datetime.datetime.utcfromtimestamp(ts[0])} ~ "
          f"{datetime.datetime.utcfromtimestamp(ts[-1])}")

    TOPK = 3
    DELAY = 1   # 延迟成交分钟数

    print(f"\n=== 反转多空(空 top{TOPK} 涨 / 多 bottom{TOPK} 跌,延迟{DELAY}min成交) ===")
    print("持有期 | 每期毛bp | 年化(毛) | 毛Sharpe | 换手率 | 临界单边成本bp")
    print("-" * 72)

    for HOLD in [15, 30, 60, 120, 240]:
        # 有效时刻 t = HOLD .. T - HOLD - DELAY - 1
        t = np.arange(HOLD, T - HOLD - DELAY)
        m = len(t)
        sig = C[t] / C[t - HOLD] - 1.0          # 信号:过去 HOLD 收益(close)
        entry = C[t + DELAY]                    # 延迟成交价
        exit_ = C[t + DELAY + HOLD]             # 出场价
        fwd = exit_ / entry - 1.0               # 未来 HOLD 收益(延迟成交口径)

        idx = np.argsort(sig, axis=1)
        fwd_sorted = np.take_along_axis(fwd, idx, axis=1)
        long_ret = fwd_sorted[:, :TOPK].mean(axis=1)
        short_ret = fwd_sorted[:, -TOPK:].mean(axis=1)
        port = long_ret - short_ret
        port = port[np.isfinite(port)]
        m2 = len(port)

        ppd = 24 * 60 / HOLD
        mean_r = float(np.mean(port))
        std_r = float(np.std(port))
        ann = (1 + mean_r) ** (ppd * 365) - 1 if mean_r > -1 else -1.0
        sharpe = mean_r / std_r * np.sqrt(ppd * 365) if std_r > 0 else 0.0

        # 换手率:相邻两期 top/bottom 集合变动比例
        chg = 0
        for i in range(1, m2):
            chg += len(set(idx[i, -TOPK:]) - set(idx[i - 1, -TOPK:])) + \
                   len(set(idx[i, :TOPK]) - set(idx[i - 1, :TOPK]))
        turnover = chg / (m2 * 2 * TOPK) if m2 > 0 else 0

        # 临界成本:每期成本 = 换手 × 2 × c,净=0 => c* = mean_r/(2*turnover)
        crit = mean_r / (2 * turnover) if turnover > 0 else np.inf
        print(f"  {HOLD:>3}min | {mean_r*10000:+8.2f} | {ann*100:+9.1f}% | "
              f"{sharpe:+7.2f} | {turnover*100:6.1f}% | {crit*10000:8.1f}")

    # 单独细看 HOLD=30 的分年
    print("\n=== HOLD=30min 分年(延迟成交) ===")
    HOLD = 30
    t = np.arange(HOLD, T - HOLD - DELAY)
    sig = C[t] / C[t - HOLD] - 1.0
    entry = C[t + DELAY]
    exit_ = C[t + DELAY + HOLD]
    fwd = exit_ / entry - 1.0
    idx = np.argsort(sig, axis=1)
    fwd_sorted = np.take_along_axis(fwd, idx, axis=1)
    port = fwd_sorted[:, :TOPK].mean(axis=1) - fwd_sorted[:, -TOPK:].mean(axis=1)
    port = port[np.isfinite(port)]
    ts_valid = ts[t][np.isfinite(fwd_sorted[:, :TOPK].mean(axis=1) - fwd_sorted[:, -TOPK:].mean(axis=1))]
    for y in [2024, 2025, 2026]:
        y0 = datetime.datetime(y, 1, 1).timestamp()
        y1 = datetime.datetime(y + 1, 1, 1).timestamp()
        mask = (ts_valid >= y0) & (ts_valid < y1)
        if mask.sum() > 0:
            sub = port[mask]
            ay = (1 + np.mean(sub)) ** (48 * 365) - 1
            print(f"  {y}: 截面 {mask.sum():,}, 毛年化 {ay*100:+.1f}%")


if __name__ == "__main__":
    main()
