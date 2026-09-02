# -*- coding: utf-8 -*-
"""清算瀑布抄底 —— 全池版回测 + 显著性检验。

BTC 单币已确认方向（-3% 插针 24h 持有 +2.42% vs 随机 -0.14%），但 n=15 太小。
本脚本扩展全池（135 币 × 2年 1h），把样本量补到能下结论：
1. 全池 -3% 插针抄底的真实期望
2. bootstrap 检验：插针抄底 vs 随机买入，差距是否显著
3. 不同深度阈值（-2%/-3%/-4%/-5%）的单调性 —— 越深越像清算瀑布，期望应越正
"""
import sqlite3
import datetime

import numpy as np

DB = "data/monitor.db"
MIN_BARS = 3000  # 至少 4 个月 1h 数据


def load_all(min_bars=MIN_BARS):
    c = sqlite3.connect(DB)
    syms = [r[0] for r in c.execute(
        "SELECT symbol, COUNT(*) n FROM klines GROUP BY symbol HAVING n >= ?", (min_bars,)
    ).fetchall()]
    data = {}
    for s in syms:
        kl = c.execute(
            f"SELECT close FROM klines WHERE symbol='{s}' ORDER BY open_time"
        ).fetchall()
        data[s] = np.array([x[0] for x in kl], dtype=np.float64)
    c.close()
    return data


def cascade_rets(data: dict, drop_thresh: float, hold_h: int, cost_side=0.0006):
    """全池识别插针，返回每笔抄底收益 logret。"""
    all_rets = []
    for s, close in data.items():
        if len(close) < 10:
            continue
        logret = np.diff(np.log(close))
        for i in range(len(logret)):
            if logret[i] <= -drop_thresh and i + 1 + hold_h < len(close):
                r = np.log(close[i + 1 + hold_h]) - np.log(close[i + 1]) - 2 * cost_side
                all_rets.append(r)
    return np.array(all_rets)


def random_baseline(data: dict, hold_h: int, n=20000, seed=42, cost_side=0.0006):
    rng = np.random.default_rng(seed)
    all_rets = []
    syms = list(data.keys())
    for _ in range(n):
        s = syms[rng.integers(len(syms))]
        close = data[s]
        if len(close) < hold_h + 5:
            continue
        start = rng.integers(0, len(close) - hold_h - 1)
        r = np.log(close[start + hold_h]) - np.log(close[start]) - 2 * cost_side
        all_rets.append(r)
    return np.array(all_rets)


def summary(rets: np.ndarray, annual_periods: float):
    n = len(rets)
    mean = rets.mean()
    std = rets.std(ddof=1) if n > 1 else 0
    sharpe = mean / std * np.sqrt(annual_periods) if std > 0 else 0
    return n, (rets > 0).sum() / n, mean, std, sharpe


def main():
    data = load_all()
    print(f"全池 {len(data)} 币, 1h K 线 2 年")
    print()

    # 1. 各深度阈值 × 持有周期的期望
    print("=" * 80)
    print(f"{'跌幅阈值':>8} {'持有':>6} {'笔数':>6} {'胜率':>7} {'单笔均值':>9} {'年化Sharpe':>10}")
    print("=" * 80)
    for thresh in [0.02, 0.03, 0.04, 0.05]:
        for hold in [4, 8, 24, 72]:
            rets = cascade_rets(data, thresh, hold)
            if len(rets) < 10:
                print(f"{thresh:>8.1%} {hold:>5}h {len(rets):>6}  (样本不足)")
                continue
            n, wr, mean, std, sh = summary(rets, 365 * 24 / hold)
            print(f"{thresh:>8.1%} {hold:>5}h {n:>6} {wr:>7.1%} {mean:>9.3%} {sh:>10.2f}")

    # 2. 对照组 + bootstrap 显著性
    print()
    print("[" + "=" * 78 + "]")
    print("显著性检验：-3% 插针抄底 vs 随机买入（bootstrap 5000 次，hold=24h）")
    cascade = cascade_rets(data, 0.03, 24)
    base = random_baseline(data, 24)
    n_c, wr_c, mean_c, _, _ = summary(cascade, 365)
    n_b, wr_b, mean_b, _, _ = summary(base, 365)
    print(f"  插针抄底: n={n_c} 胜率={wr_c:.1%} 单笔均值={mean_c:+.3%}")
    print(f"  随机买入: n={n_b} 胜率={wr_b:.1%} 单笔均值={mean_b:+.3%}")
    print(f"  差值 = {mean_c - mean_b:+.3%}")

    # bootstrap：从随机对照组抽 n_c 个，看多少次均值 >= cascade 均值
    rng = np.random.default_rng(7)
    exceeds = 0
    B = 5000
    for _ in range(B):
        sample = rng.choice(base, size=n_c, replace=True)
        if sample.mean() >= mean_c:
            exceeds += 1
    p_value = exceeds / B
    print(f"  bootstrap p 值 = {p_value:.4f} (随机买入均值 >= 插针抄底均值的概率)")
    print(f"  → {'显著（alpha 真实）' if p_value < 0.05 else '不显著（可能是运气）'}")

    # 3. 深度单调性
    print()
    print("深度单调性（24h 持有，越深越应像清算瀑布）：")
    for thresh in [0.02, 0.03, 0.04, 0.05]:
        rets = cascade_rets(data, thresh, 24)
        if len(rets) >= 5:
            n, wr, mean, _, _ = summary(rets, 365)
            print(f"  -{thresh:.0%} 插针: n={n:>5} 胜率={wr:.1%} 单笔={mean:+.3%}")


if __name__ == "__main__":
    main()
