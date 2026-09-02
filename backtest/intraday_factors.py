"""
1m 精细因子 vs 1h 粗因子 —— 截面 IC 对比,验证"1m 数据的正确用法=精细特征工程"。

因子(每 1h 截面,16 币,预测未来 96h 收益):
  mom_720    过去 30 天收益(慢动量,1h 粗粒度基准)
  mom_24h    过去 24h 收益(日内动量,1m 精确)
  rev_1h     过去 1h 收益(短反转,1m 精确)
  rv_24h     过去 24h 的 1m realized vol(1m 独有的精细波动率)
  rev_24h    过去 24h 的 15min 反转强度(1m 独有的日内反转结构)
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


def cross_ic_rank(a, b):
    """逐行秩相关 IC(a,b 为 (H,N) 含 NaN)。"""
    H, N = a.shape
    out = []
    for i in range(H):
        ai, bi = a[i], b[i]
        m = np.isfinite(ai) & np.isfinite(bi)
        if m.sum() < 8:
            continue
        ra = pd.Series(ai[m]).rank().values
        rb = pd.Series(bi[m]).rank().values
        if ra.std() == 0 or rb.std() == 0:
            continue
        out.append(np.corrcoef(ra, rb)[0, 1])
    return np.array(out)


def main():
    C, ts, symbols = load_1m()
    T, N = C.shape
    print(f"1m 矩阵 {T:,} × {N} 币")
    print(f"{datetime.datetime.utcfromtimestamp(ts[0])} ~ "
          f"{datetime.datetime.utcfromtimestamp(ts[-1])}")

    # 1h 截面(整点)
    hour_idx = np.where((ts - ts[0]) % 3600 == 0)[0]
    H = len(hour_idx)
    print(f"1h 截面数: {H:,}")

    FUT_H = 96   # 未来 96h
    FUT_M = FUT_H * 60

    # 预计算每 1h 截面未来 96h 收益(用 1m close)
    fwd = np.full((H, N), np.nan)
    for k, i in enumerate(hour_idx):
        j = i + FUT_M
        if j < T:
            fwd[k] = C[j] / C[i] - 1.0

    # 因子
    def factor_past(k, minutes):
        i = hour_idx[k]
        return C[i] / C[i - minutes] - 1.0

    mom_720 = np.full((H, N), np.nan)
    mom_24h = np.full((H, N), np.nan)
    rev_1h = np.full((H, N), np.nan)
    rv_24h = np.full((H, N), np.nan)
    rev_24h = np.full((H, N), np.nan)

    for k in range(H):
        i = hour_idx[k]
        if i < 720 * 60:
            continue
        mom_720[k] = C[i] / C[i - 720 * 60] - 1.0
        mom_24h[k] = C[i] / C[i - 24 * 60] - 1.0
        rev_1h[k] = C[i] / C[i - 60] - 1.0
        # rv_24h:过去24h的1m收益std
        seg = C[i - 24 * 60:i + 1]
        lr = np.log(seg[1:] / seg[:-1])
        rv_24h[k] = np.nanstd(lr, axis=0) * np.sqrt(1440)
        # rev_24h:过去24h的15min反转强度 = -corr(过去15min收益, 未来15min收益) 的累计
        # 简化:过去24h内 sum(过去15min收益 * 未来15min收益) 为负 => 反转强
        seg_l = C[i - 24 * 60:i + 1]
        r15 = seg_l[15:] / seg_l[:-15] - 1.0
        f15 = seg_l[30:] / seg_l[15:-15] - 1.0
        nmin = min(len(r15), len(f15))
        rev_24h[k] = -(r15[:nmin] * f15[:nmin]).sum(axis=0) / nmin

    # 截断:去掉前 720*60 分钟的 warmup,去掉未来 96h 不足的尾部
    valid = (np.arange(H) >= 720) & np.isfinite(fwd).any(axis=1)
    print(f"有效截面(去 warmup+尾部): {valid.sum():,}")

    factors = {
        "mom_720(30天,基准)": mom_720,
        "mom_24h(日内动量)": mom_24h,
        "rev_1h(短反转)": rev_1h,
        "rv_24h(1m已实现波动)": rv_24h,
        "rev_24h(日内反转强度)": rev_24h,
    }
    print("\n=== 各因子 vs 未来96h收益 截面 IC ===")
    print(f"{'因子':<22} {'IC均值':>10} {'IC年化':>10} {'IC>0占比':>10} {'t值':>8}")
    for name, F in factors.items():
        ic = cross_ic_rank(F[valid], fwd[valid])
        ic = ic[np.isfinite(ic)]
        mean_ic = float(np.mean(ic))
        t = mean_ic / (np.std(ic) / np.sqrt(len(ic))) if len(ic) else 0
        pos = float((ic > 0).mean())
        print(f"{name:<22} {mean_ic:>+10.4f} {mean_ic*96**0.5*16:>+10.3f} "
              f"{pos*100:>9.1f}% {t:>+8.1f}")


if __name__ == "__main__":
    main()
