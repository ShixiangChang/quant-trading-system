"""
诊断:rv_24h 低波动异象的方向为什么在 24h 回测里反了。
算 rv_24h 在不同持有期(24/48/96/192h)的 IC + top5/bottom5 收益,定位方向翻转点。
"""
import sqlite3
import numpy as np
import pandas as pd
import datetime

DB = "data/monitor.db"
MIN_ROWS = 900_000


def load_1m():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol",
                           values="close", aggfunc="last").sort_index()[symbols]
    return close.values.astype(float), close.index.values.astype(np.int64)


def main():
    C, ts = load_1m()
    T, N = C.shape
    DAY_M = 24 * 60
    t = np.arange(DAY_M, T - 8 * DAY_M, DAY_M)   # 每天一个截面,留 8 天未来
    m = len(t)

    # rv_24h
    rv = np.full((m, N), np.nan)
    for k, i in enumerate(t):
        seg = C[i - DAY_M:i + 1]
        lr = np.log(seg[1:] / seg[:-1])
        rv[k] = np.nanstd(lr, axis=0) * np.sqrt(1440)

    print("持有期 | rv_24h IC | 多低波动5 日均bp | 空高波动5 日均bp | 多空组合bp")
    print("-" * 70)
    for HOLD_H in [24, 48, 96, 192]:
        HOLD_M = HOLD_H * 60
        fwd = np.full((m, N), np.nan)
        for k, i in enumerate(t):
            if i + HOLD_M < T:
                fwd[k] = C[i + HOLD_M] / C[i] - 1.0
        # IC
        ics = []
        for k in range(m):
            a, b = rv[k], fwd[k]
            mm = np.isfinite(a) & np.isfinite(b)
            if mm.sum() < 8:
                continue
            ra = pd.Series(a[mm]).rank().values
            rb = pd.Series(b[mm]).rank().values
            if ra.std() == 0 or rb.std() == 0:
                continue
            ics.append(np.corrcoef(ra, rb)[0, 1])
        ic = np.mean(ics)

        idx = np.argsort(rv, axis=1)
        fs = np.take_along_axis(fwd, idx, axis=1)
        long_r = np.nanmean(fs[:, :5], axis=1)
        short_r = np.nanmean(fs[:, -5:], axis=1)
        port = long_r - short_r
        print(f"{HOLD_H:>4}h  | {ic:>+8.4f} | {np.nanmean(long_r)*10000:>+13.2f} | "
              f"{np.nanmean(short_r)*10000:>+13.2f} | {np.nanmean(port)*10000:>+9.2f}")

    # 还要看:rv 本身是否等于"过去24h动量绝对值"——高波动是否=大涨或大跌
    print("\n诊断:rv_24h 与 过去24h收益的关系(高波动=大涨还是大跌?)")
    mom24 = np.full((m, N), np.nan)
    for k, i in enumerate(t):
        mom24[k] = C[i] / C[i - DAY_M] - 1.0
    # 高波动组的过去24h平均收益 vs 低波动组
    idx = np.argsort(rv, axis=1)
    ms = np.take_along_axis(mom24, idx, axis=1)
    print(f"  低波动5币 过去24h平均收益: {np.nanmean(ms[:, :5])*100:+.2f}%")
    print(f"  高波动5币 过去24h平均收益: {np.nanmean(ms[:, -5:])*100:+.2f}%")


if __name__ == "__main__":
    main()
