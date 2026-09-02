"""
低波动异象腿(lovol)—— rv_24h 多空回测,验证 1m 挖出的最强因子能否交易。

策略:每 24h 截面,按过去24h的1m已实现波动率(rv_24h)排序,
做多最低波动 5 币、做空最高波动 5 币,持有 24h,延迟1min成交。
"""
import sqlite3
import numpy as np
import pandas as pd
import datetime

DB = "data/monitor.db"
MIN_ROWS = 900_000


def load_1m():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol",
                           values="close", aggfunc="last").sort_index()[symbols]
    return close.values.astype(float), close.index.values.astype(np.int64), symbols


def main():
    C, ts, symbols = load_1m()
    T, N = C.shape
    print(f"1m 矩阵 {T:,} × {N} 币, {datetime.datetime.utcfromtimestamp(ts[0])} ~ "
          f"{datetime.datetime.utcfromtimestamp(ts[-1])}")

    HOLD_H = 24          # 持有 24h
    TOPK = 5             # 多空各 5 币
    HOLD_M = HOLD_H * 60
    DAY_M = 24 * 60

    # 每 24h 一个截面(UTC 0 点),t 时刻算 rv_24h,做多低波动/做空高波动,持有 24h
    t = np.arange(DAY_M, T - HOLD_M - 1, DAY_M)   # 每天一个调仓点
    m = len(t)
    print(f"调仓截面数: {m:,} (每{HOLD_H}h)")

    rv = np.full((m, N), np.nan)
    for k, i in enumerate(t):
        seg = C[i - DAY_M:i + 1]
        lr = np.log(seg[1:] / seg[:-1])
        rv[k] = np.nanstd(lr, axis=0) * np.sqrt(1440)

    # 延迟成交:入场 t+1min close,出场 t+1min+HOLD close
    entry = C[t + 1]
    exit_ = C[t + 1 + HOLD_M]
    fwd = exit_ / entry - 1.0        # (m, N)

    idx = np.argsort(rv, axis=1)     # 升序:低波动在前
    fwd_sorted = np.take_along_axis(fwd, idx, axis=1)
    long_ret = fwd_sorted[:, :TOPK].mean(axis=1)   # 做多低波动
    short_ret = fwd_sorted[:, -TOPK:].mean(axis=1) # 做空高波动
    port = long_ret - short_ret
    port = port[np.isfinite(port)]

    mean_r = float(np.mean(port))
    std_r = float(np.std(port))
    ann = (1 + mean_r) ** 365 - 1
    sharpe = mean_r / std_r * np.sqrt(365) if std_r > 0 else 0

    # 换手率:相邻两期 top/bottom 集合变动
    chg = 0
    for i in range(1, m):
        chg += len(set(idx[i, :TOPK]) - set(idx[i - 1, :TOPK])) + \
               len(set(idx[i, -TOPK:]) - set(idx[i - 1, -TOPK:]))
    turnover = chg / (m * 2 * TOPK)
    crit = mean_r / (2 * turnover) if turnover > 0 else np.inf

    print(f"\n=== 低波动多空(多最低波动{TOPK} / 空最高波动{TOPK},持有{HOLD_H}h) ===")
    print(f"每日毛收益 {mean_r*10000:+.2f} bp  毛年化 {ann*100:+.1f}%  毛Sharpe {sharpe:.2f}")
    print(f"换手率 {turnover*100:.1f}%  临界单边成本 {crit*10000:.1f} bp")
    for c in [0.0002, 0.0005, 0.0010]:
        net = mean_r - 2 * turnover * c
        ny = (1 + net) ** 365 - 1 if net > -1 else -1
        print(f"  单边 {c*10000:.0f}bp → 净每日 {net*10000:+.2f}bp, 净年化 {ny*100:+.1f}%")

    # 净值
    eq = np.cumprod(1 + port)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    print(f"毛净值(2年): {eq[-1]:.2f}x  最大回撤 {dd.min()*100:.1f}%")

    # 分年
    ts_valid = ts[t][np.isfinite(long_ret - short_ret)]
    port_valid = port
    print("分年毛年化:")
    for y in [2024, 2025, 2026]:
        y0 = datetime.datetime(y, 1, 1).timestamp()
        y1 = datetime.datetime(y + 1, 1, 1).timestamp()
        mask = (ts_valid >= y0) & (ts_valid < y1)
        if mask.sum() > 0:
            ay = (1 + np.mean(port_valid[mask])) ** 365 - 1
            print(f"  {y}: {mask.sum():,} 天, 毛年化 {ay*100:+.1f}%")


if __name__ == "__main__":
    main()
