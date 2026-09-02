"""
日内反转增强(向量化)—— 用 volume/波动率做条件,看能否做厚 edge 突破成本墙。
"""
import sqlite3
import numpy as np
import pandas as pd
import datetime

DB = "data/monitor.db"
MIN_ROWS = 900_000


def load():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT symbol, open_time, close, volume FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol",
                           values="close", aggfunc="last").sort_index()[symbols]
    vol = df.pivot_table(index="open_time", columns="symbol",
                         values="volume", aggfunc="sum").sort_index()[symbols]
    return close.values.astype(float), vol.values.astype(float), close.index.values.astype(np.int64)


def backtest_vec(sig, fwd, mask_group, topk=3):
    """向量化多空:做多 sig 最小的 topk、做空 sig 最大的 topk(仅限 mask_group 内)。"""
    s = np.where(mask_group, sig, np.nan)
    f = np.where(mask_group, fwd, np.nan)
    valid = np.sum(mask_group, axis=1)
    ok = valid >= 2 * topk
    s = s[ok]; f = f[ok]
    oa = np.argsort(s, axis=1)          # 升序,NaN 排最后
    od = np.argsort(-s, axis=1)         # 降序,NaN 排最后
    long_ret = np.take_along_axis(f, oa[:, :topk], axis=1).mean(axis=1)
    short_ret = np.take_along_axis(f, od[:, :topk], axis=1).mean(axis=1)
    port = long_ret - short_ret
    port = port[np.isfinite(port)]
    return port


def main():
    C, V, ts = load()
    T, N = C.shape
    print(f"矩阵 {T:,} 分钟 × {N} 币, {datetime.datetime.utcfromtimestamp(ts[0])} ~ "
          f"{datetime.datetime.utcfromtimestamp(ts[-1])}")

    HOLD, TOPK, DELAY = 15, 3, 1
    t = np.arange(HOLD, T - HOLD - DELAY)
    sig = C[t] / C[t - HOLD] - 1.0
    entry = C[t + DELAY]
    exit_ = C[t + DELAY + HOLD]
    fwd = exit_ / entry - 1.0

    # 过去15min累计成交量 + 波动幅度
    vsum = np.zeros((len(t), N))
    for lag in range(HOLD):
        vsum += V[t - lag]
    vol_amp = np.abs(sig)

    thr_v = np.nanmedian(vol_amp, axis=1, keepdims=True)
    thr_q = np.nanmedian(vsum, axis=1, keepdims=True)

    groups = {
        "全部币": np.ones_like(sig, dtype=bool),
        "高波动(|ret|>中位)": vol_amp > thr_v,
        "高量(vol>中位)": vsum > thr_q,
        "高波动+高量": (vol_amp > thr_v) & (vsum > thr_q),
        "低波动(<=中位)": vol_amp <= thr_v,
        "低量(<=中位)": vsum <= thr_q,
    }

    print("\n=== 反转增强对比(HOLD=15min, 延迟1min, 多空top3) ===")
    print(f"{'分组':<22} {'每期毛bp':>10} {'临界成本bp':>11} {'截面数':>10} {'毛年化':>10}")
    results = []
    for name, mask in groups.items():
        port = backtest_vec(sig, fwd, mask, TOPK)
        if len(port) == 0:
            continue
        mean_r = float(np.mean(port))
        std_r = float(np.std(port))
        ppd = 96
        ann = (1 + mean_r) ** (ppd * 365) - 1 if mean_r > -1 else -1.0
        # 换手率(粗略 25%)
        turn = 0.25
        crit = mean_r / (2 * turn) if turn > 0 else np.inf
        print(f"{name:<22} {mean_r*10000:>+10.2f} {crit*10000:>11.1f} "
              f"{len(port):>10,} {ann*100:>+10.1f}%")
        results.append((name, mean_r, crit, len(port)))

    # 找最优
    best = max(results, key=lambda x: x[1])
    print(f"\n最优分组: {best[0]}  每期 {best[1]*10000:+.2f}bp  临界成本 {best[2]*10000:.1f}bp")


if __name__ == "__main__":
    main()
