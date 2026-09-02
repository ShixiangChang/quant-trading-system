# -*- coding: utf-8 -*-
"""资金费率极端反向回测：费率极正→空、极负→多，预期均值回归。

核心假设：
资金费率反映「谁被迫付钱」。费率极正 = 多头极度拥挤（多头付空头钱），
对手方被迫成交，价格后续倾向回调 → 反向做空。
费率极负 = 空头极度拥挤（空头付多头钱）→ 反向做多，预期反弹。

数据：funding_hist（8h 结算）+ klines（1h），BTC 先做（最干净、费率最连续）。
扫多组阈值 × 持有周期，观察哪格稳定正收益。
"""
import sqlite3
import datetime

import numpy as np

DB = "data/monitor.db"
SYMBOL = "BTCUSDT"


def load():
    c = sqlite3.connect(DB)
    fr = c.execute(
        f"SELECT funding_time, funding FROM funding_hist WHERE symbol='{SYMBOL}' ORDER BY funding_time"
    ).fetchall()
    kl = c.execute(
        f"SELECT open_time, close FROM klines WHERE symbol='{SYMBOL}' ORDER BY open_time"
    ).fetchall()
    c.close()

    fr_ts = np.array([x[0] for x in fr], dtype=np.int64)
    fr_val = np.array([x[1] for x in fr], dtype=np.float64)
    kl_ts = np.array([x[0] for x in kl], dtype=np.int64)
    kl_close = np.array([x[1] for x in kl], dtype=np.float64)

    # klines 1h 收盘价 → 用 searchsorted 快速定位任意时间点的价格
    price_at = lambda t: kl_close[np.searchsorted(kl_ts, t, side="right") - 1]
    return fr_ts, fr_val, kl_ts, kl_close, price_at


def backtest(mad_thresh: float, hold_8h: int, cost_side: float = 0.0006) -> dict:
    """在费率 |z| > thresh 时反向开仓，持 hold_8h 个周期，统计收益。

    mad_thresh: 以 MAD（中位数绝对偏差，稳健）衡量的偏离倍数
    hold_8h: 持有周期数（每周期 8h）
    收益口径：logret × 方向，扣双边手续费。
    """
    fr_ts, fr_val, kl_ts, kl_close, price_at = load()

    # 稳健阈值：中位数 + k*MAD
    med = np.median(fr_val)
    mad = np.median(np.abs(fr_val - med))
    if mad == 0:
        return None
    z = (fr_val - med) / mad

    entry_times = []
    directions = []
    for i in range(len(fr_ts) - 1):
        if z[i] > mad_thresh:      # 极正 → 做空
            entry_times.append(fr_ts[i])
            directions.append(-1.0)
        elif z[i] < -mad_thresh:   # 极负 → 做多
            entry_times.append(fr_ts[i])
            directions.append(1.0)

    entry_times = np.array(entry_times, dtype=np.int64)
    directions = np.array(directions)

    if len(entry_times) == 0:
        return None

    # 每笔仓位的进入/退出价
    rets = []
    for t, d in zip(entry_times, directions):
        t0 = int(t)
        exit_t = t0 + hold_8h * 8 * 3600
        # 退出价用 searchsorted 定位，越界则取最后价
        idx = np.searchsorted(kl_ts, exit_t, side="right") - 1
        idx = min(idx, len(kl_close) - 1)
        p0 = price_at(t0)
        p1 = kl_close[idx]
        if p0 <= 0 or p1 <= 0:
            continue
        rets.append(d * (np.log(p1) - np.log(p0)) - 2 * cost_side)

    rets = np.array(rets)
    n = len(rets)
    wins = (rets > 0).sum()
    mean = rets.mean()
    std = rets.std(ddof=1) if n > 1 else 0
    sharpe = mean / std * np.sqrt(365 / (hold_8h / 3)) if std > 0 else 0
    return {
        "thresh": mad_thresh, "hold_8h": hold_8h, "n": n,
        "win_rate": wins / n, "mean": mean, "std": std,
        "sharpe": sharpe, "total": rets.sum(),
    }


def main():
    fr_ts, fr_val, _, _, _ = load()
    med = np.median(fr_val)
    mad = np.median(np.abs(fr_val - med))
    print(f"BTC funding 2年: {len(fr_val)} 个结算点")
    print(f"  中位数 {med:.5%}, MAD {mad:.5%}")
    print(f"  时间 {datetime.datetime.fromtimestamp(fr_ts[0])} -> {datetime.datetime.fromtimestamp(fr_ts[-1])}")
    print()

    # 扫阈值 × 持有周期
    print("=" * 78)
    print(f"{'阈值(MAD)':>10} {'持有':>6} {'笔数':>5} {'胜率':>7} {'单笔均值':>9} {'总收益':>8} {'年化Sharpe':>10}")
    print("=" * 78)
    for thresh in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for hold in [1, 3, 6, 12, 24]:  # 8h / 1d / 2d / 4d / 8d
            r = backtest(thresh, hold)
            if r is None or r["n"] < 5:
                print(f"{thresh:>10} {hold*8:>5}h {r['n'] if r else 0:>5}  (样本不足)")
                continue
            print(f"{thresh:>10} {hold*8:>5}h {r['n']:>5} {r['win_rate']:>7.1%} "
                  f"{r['mean']:>9.3%} {r['total']:>8.2%} {r['sharpe']:>10.2f}")


if __name__ == "__main__":
    main()
