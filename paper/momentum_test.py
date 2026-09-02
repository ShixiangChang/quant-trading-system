# -*- coding: utf-8 -*-
"""横截面动量检验：过去 N 天强势的币，未来是否继续强？

背景：资金费率反向、清算瀑布抄底两条「反极端」alpha 均被全池数据证伪，
共同规律是「极端之后是动量不是反转」。本脚本正面验证动量假设：
强者是否恒强（cross-sectional momentum，Jegadeesh-Titman 1993）。

方法：
- 每个横截面日 t，算每币过去 N 天 logret（skip 最近 1 天防微观结构噪声）
- 排序分 5 档（quintile），Q5=最强势，Q1=最弱
- 记未来 H 天的收益，统计各档均值、Q5-Q1 多空价差
- 若 Q5 均值 > Q1 均值 且单调 → 动量成立，之前 cs 模型方向搞反了
"""
import sqlite3
import datetime

import numpy as np

DB = "data/monitor.db"
MIN_BARS = 3000


def load_all():
    c = sqlite3.connect(DB)
    syms = [r[0] for r in c.execute(
        "SELECT symbol, COUNT(*) n FROM klines GROUP BY symbol HAVING n >= ?", (MIN_BARS,)
    ).fetchall()]
    # 对齐到统一的 open_time 时间轴（用众数最多的 BTC 时间点）
    data = {}
    for s in syms:
        kl = c.execute(
            f"SELECT open_time, close FROM klines WHERE symbol='{s}' ORDER BY open_time"
        ).fetchall()
        data[s] = {"ts": np.array([x[0] for x in kl], dtype=np.int64),
                   "close": np.array([x[1] for x in kl], dtype=np.float64)}
    return data


def build_panel(data, freq_h=24):
    """落成 panel：每个 rebalance 时刻 × 每币的 close。用 BTC 时间轴。"""
    btc = data["BTCUSDT"]
    # 取 BTC 的日级时间点（每 24h 一个），作为横截面
    ts_all = btc["ts"]
    step = freq_h * 3600
    rebal_ts = ts_all[::freq_h]
    panel = {}
    for s, d in data.items():
        # 用 searchsorted 对齐每币价格到 rebal_ts
        idx = np.searchsorted(d["ts"], rebal_ts, side="right") - 1
        idx = np.clip(idx, 0, len(d["close"]) - 1)
        panel[s] = d["close"][idx]
    return rebal_ts, panel


def momentum_test(lookback_h=24 * 7, hold_h=24 * 7, skip_h=24, n_q=5):
    data = load_all()
    rebal_ts, panel = build_panel(data, freq_h=24)

    syms = list(panel.keys())
    n_t = len(rebal_ts)
    closes = np.array([panel[s] for s in syms])  # (n_sym, n_t)

    # 每日收益（logret）
    logret = np.log(closes[:, 1:]) - np.log(closes[:, :-1])

    # 回看窗口的收益（过去 lookback 根 K 线，skip 最近 skip 根）
    lookback_n = lookback_h // 24
    skip_n = skip_h // 24
    hold_n = hold_h // 24

    quintile_fwd = [[] for _ in range(n_q)]
    ls_spread = []  # Q5 - Q1

    for t in range(lookback_n + skip_n, n_t - hold_n):
        # 过去 lookback_n 天的累计收益（skip 最近 skip_n 天）
        past = logret[:, t - lookback_n - skip_n: t - skip_n].sum(axis=1)
        past = np.where(np.isfinite(past), past, 0)
        # 未来 hold_n 天收益
        fwd = logret[:, t: t + hold_n].sum(axis=1)
        fwd = np.where(np.isfinite(fwd), fwd, 0)

        # 排序分档
        order = np.argsort(past)
        q_size = len(syms) // n_q
        for q in range(n_q):
            idx = order[q * q_size: (q + 1) * q_size]
            quintile_fwd[q].extend(fwd[idx].tolist())
        ls_spread.append(fwd[order[-q_size:]].mean() - fwd[order[:q_size]].mean())

    quintile_means = [np.mean(q) if q else 0 for q in quintile_fwd]
    ls = np.array(ls_spread)
    return quintile_means, ls


def main():
    print("横截面动量检验（全池 135 币 × 2 年 1h）")
    print("=" * 70)
    for lookback in [24, 24 * 3, 24 * 7, 24 * 30]:
        q_means, ls = momentum_test(lookback_h=lookback, hold_h=24 * 7)
        ls_mean = ls.mean() if len(ls) else 0
        ls_std = ls.std(ddof=1) if len(ls) > 1 else 0
        t_stat = ls_mean / ls_std * np.sqrt(len(ls)) if ls_std > 0 else 0
        n_rebal = len(ls)
        print(f"\n回看 {lookback // 24:>3} 天, 持有 7 天 (rebalance 次数 {n_rebal}):")
        labels = ["Q1 最弱", "Q2", "Q3", "Q4", "Q5 最强"]
        for l, m in zip(labels, q_means):
            print(f"   {l}: 未来7天均值 {m:+.3%}")
        print(f"   Q5-Q1 多空价差: 每期 {ls_mean:+.3%}  (t={t_stat:.2f})  "
              f"{'✓动量成立' if ls_mean > 0 else '✗动量不成立/反向'}")


if __name__ == "__main__":
    main()
