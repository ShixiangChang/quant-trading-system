# -*- coding: utf-8 -*-
"""清算瀑布抄底回测：暴跌插针后回抽，捕捉「对手方被迫成交」的超跌反弹。

核心假设（与资金费率反向的本质区别）：
清算瀑布 = 大量杠杆多头被强平引擎「强制」平仓（被迫成交，不是主动加杠杆），
价格插针超跌 → 随后回抽填补缺口。抄这个底吃的是回抽。

信号（纯价格指纹，不依赖强平流水，回避 Binance 只给 30 天 OI 的限制）：
单根/连续 1h 暴跌超过阈值 → 判定清算瀑布插针。

回测：
- 插针收盘买入，持 N 根 1h 后平仓
- 扫多档跌幅阈值 × 持有周期
- 对照组：随机时点买入同持有时长，若两者无差异 → 插针抄底无 alpha
"""
import sqlite3
import datetime

import numpy as np

DB = "data/monitor.db"
SYMBOL = "BTCUSDT"


def load():
    c = sqlite3.connect(DB)
    kl = c.execute(
        f"SELECT open_time, close, volume, quote_volume FROM klines "
        f"WHERE symbol='{SYMBOL}' ORDER BY open_time"
    ).fetchall()
    c.close()
    ts = np.array([x[0] for x in kl], dtype=np.int64)
    close = np.array([x[1] for x in kl], dtype=np.float64)
    return ts, close


def backtest(drop_thresh: float, hold_h: int, cost_side: float = 0.0006) -> dict:
    ts, close = load()
    logret = np.diff(np.log(close))

    # 插针事件点：1h 跌幅 <= -thresh（注意：ret[i] 对应 close[i] -> close[i+1]）
    signals = []
    for i in range(len(logret)):
        if logret[i] <= -drop_thresh:
            # 在 close[i+1]（插针后的收盘）买入
            signals.append(i + 1)

    if not signals:
        return None

    rets = []
    for s in signals:
        exit_s = s + hold_h
        if exit_s >= len(close):
            continue
        r = np.log(close[exit_s]) - np.log(close[s]) - 2 * cost_side
        rets.append(r)

    rets = np.array(rets)
    if len(rets) == 0:
        return None
    n = len(rets)
    wins = (rets > 0).sum()
    mean = rets.mean()
    std = rets.std(ddof=1) if n > 1 else 0
    sharpe = mean / std * np.sqrt(365 * 24 / hold_h) if std > 0 else 0
    return {"n": n, "win_rate": wins / n, "mean": mean, "total": rets.sum(), "sharpe": sharpe}


def random_baseline(hold_h: int, n_sample: int = 2000, cost_side: float = 0.0006) -> dict:
    """对照组：随机时点买入持 hold_h，看插针抄底是否跑赢随机。"""
    ts, close = load()
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(close) - hold_h - 1, n_sample)
    rets = []
    for s in starts:
        r = np.log(close[s + hold_h]) - np.log(close[s]) - 2 * cost_side
        rets.append(r)
    rets = np.array(rets)
    mean = rets.mean()
    std = rets.std(ddof=1)
    sharpe = mean / std * np.sqrt(365 * 24 / hold_h) if std > 0 else 0
    return {"n": len(rets), "win_rate": (rets > 0).sum() / len(rets), "mean": mean, "sharpe": sharpe}


def main():
    ts, close = load()
    print(f"BTC 1h 数据: {len(close)} 根, {datetime.datetime.fromtimestamp(ts[0])} -> {datetime.datetime.fromtimestamp(ts[-1])}")
    print()

    print("=" * 76)
    print(f"{'跌幅阈值':>8} {'持有':>6} {'笔数':>5} {'胜率':>7} {'单笔均值':>9} {'总收益':>8} {'年化Sharpe':>10}")
    print("=" * 76)
    for thresh in [0.02, 0.03, 0.04, 0.05]:
        for hold in [1, 4, 8, 24, 72]:  # 1h / 4h / 8h / 1d / 3d
            r = backtest(thresh, hold)
            if r is None or r["n"] < 5:
                print(f"{thresh:>8.1%} {hold:>5}h {r['n'] if r else 0:>5}  (样本不足)")
                continue
            print(f"{thresh:>8.1%} {hold:>5}h {r['n']:>5} {r['win_rate']:>7.1%} "
                  f"{r['mean']:>9.3%} {r['total']:>8.2%} {r['sharpe']:>10.2f}")

    print()
    print("对照组（随机时点买入，同持有时长）：")
    for hold in [4, 8, 24, 72]:
        b = random_baseline(hold)
        print(f"  持有 {hold:>3}h: 单笔均值 {b['mean']:+.3%}  胜率 {b['win_rate']:.1%}  年化Sharpe {b['sharpe']:.2f}")


if __name__ == "__main__":
    main()
