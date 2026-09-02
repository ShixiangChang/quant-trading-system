"""
日内短周期动量探索 —— 1m 数据截面 IC 网格(向量化版)。

量级:12 币 × 2 年 × 1440 分钟 ≈ 1260 万分钟截面(日频 12 万截面的 100 倍)。
试金石:过去 L 分钟收益 vs 未来 h 分钟收益的截面秩相关 IC。
"""
import sqlite3
import numpy as np
import pandas as pd
import datetime

DB = "data/monitor.db"
MIN_ROWS = 900_000
LOOKBACKS = [15, 30, 60, 120, 240, 480, 720]
HORIZONS = [15, 30, 60, 120, 240, 480, 720]


def row_rank(x):
    """逐行 rank (1-based),处理 NaN:NaN 排最后,rank 后仍保留 NaN 位置。"""
    r = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        row = x[i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        order = np.argsort(np.argsort(row[mask]))
        r[i, mask] = order.astype(float) + 1.0
    return r


def cross_ic(a, b):
    """逐行秩相关 IC。a,b 为 (T, N),含 NaN。"""
    ra = row_rank(a)
    rb = row_rank(b)
    ra_c = ra - np.nanmean(ra, axis=1, keepdims=True)
    rb_c = rb - np.nanmean(rb, axis=1, keepdims=True)
    num = np.nansum(ra_c * rb_c, axis=1)
    den = np.sqrt(np.nansum(ra_c ** 2, axis=1) * np.nansum(rb_c ** 2, axis=1))
    ic = num / den
    return ic


def main():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]

    print(f"币数 {len(symbols)}, 总行数 {len(df):,}")

    # pivot 成 (T, N) close 矩阵
    close = df.pivot_table(index="open_time", columns="symbol",
                           values="close", aggfunc="last").sort_index()
    close = close[symbols]
    T, N = close.shape
    ts = close.index.values.astype(np.int64)
    print(f"close 矩阵: {T:,} 分钟 × {N} 币")
    print(f"时间: {datetime.datetime.utcfromtimestamp(ts[0])} ~ "
          f"{datetime.datetime.utcfromtimestamp(ts[-1])}")

    C = close.values.astype(float)   # (T, N)
    # 收益:1m 收益率
    ret = C[1:] / C[:-1] - 1.0       # (T-1, N)

    # 降采样:每 30min 一个截面(减少重叠自相关)
    step = 30
    print(f"降采样每 {step}min 一个截面,约 {T // step:,} 截面\n")

    print("截面 IC 矩阵 (行=过去L分钟收益, 列=未来h分钟收益):")
    hdr = "past\\fwd   " + "".join(f"{h:>9}" for h in HORIZONS)
    print(hdr)
    # 累计对数收益 cs[t] = sum_{i< t} logret[i],即从起点到 t 时刻前的累计收益
    logret = np.log(C[1:] / C[:-1])          # (T-1, N)
    cs = np.zeros((T, N))
    cs[1:] = np.cumsum(logret, axis=0)       # cs[t] = 0..t-1 的累计对数收益
    cs = np.where(np.isfinite(cs), cs, 0.0)  # NaN 置 0 防扩散

    best = []
    for L in LOOKBACKS:
        # past[t] = 过去 L 分钟累计对数收益 = cs[t] - cs[t-L]
        past = np.full((T, N), np.nan)
        past[L:] = cs[L:] - cs[:T - L]

        line = f"{L:>5}min "
        for h in HORIZONS:
            # 未来 h 分钟收益: fwd[t] = cs[t+h] - cs[t]
            fwd = np.full((T, N), np.nan)
            fwd[:T - h] = cs[h:] - cs[:T - h]
            # 降采样 + 去掉头尾 NaN
            sub = past[::step]
            sub_f = fwd[::step]
            ic = cross_ic(sub, sub_f)
            ic_finite = ic[np.isfinite(ic)]
            mean_ic = float(np.nanmean(ic)) if len(ic_finite) > 0 else np.nan
            line += f"{mean_ic:>+9.4f}"
            best.append((L, h, mean_ic, len(ic_finite)))
        print(line)

    flat = sorted([b for b in best if np.isfinite(b[2])],
                  key=lambda x: -abs(x[2]))
    print("\nTop 10 |IC| 组合:")
    for L, h, ic, n in flat[:10]:
        print(f"  past_{L:>4}m -> fwd_{h:>4}m : IC={ic:+.4f} (n截面={n:,})")


if __name__ == "__main__":
    main()
