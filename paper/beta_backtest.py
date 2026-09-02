# -*- coding: utf-8 -*-
"""纯 beta 趋势跟随回测：全池等权指数 vs 慢均线，牛持多 / 熊空仓。

目标：证明「至少能拿 beta」——完全不碰截面模型（cs 排序已被证伪是负的），
只做趋势择时：指数在均线上方就持有等权多头吃上涨，跌到均线下方就空仓躲下跌。
这是 Faber 2007 / CTA 行业最基础的 beta 策略，方向跟随、不预测。
"""
import glob
import os

import numpy as np
import pandas as pd

from model import config as c

POOL = (c.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()


def load_equal_weight_index() -> pd.Series:
    """全池等权指数收益（每个时间截面所有币 logret 的均值）。"""
    frames = []
    for sym in POOL:
        hits = glob.glob(str(c.CACHE_DIR / f"klines_*d_{sym}.csv"))
        if not hits:
            continue
        df = pd.read_csv(hits[0], usecols=["open_time", "close"])
        df["symbol"] = sym
        frames.append(df)
    kl = pd.concat(frames, ignore_index=True)
    kl = kl.sort_values("open_time")
    kl["logret"] = kl.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))
    idx_ret = kl.groupby("open_time")["logret"].mean().sort_index()
    return idx_ret.dropna()


def backtest(ma_hours: int, cost_side: float = 0.0006) -> dict:
    idx_ret = load_equal_weight_index()
    idx_price = np.exp(idx_ret.cumsum())
    ma = idx_price.rolling(ma_hours).mean()

    signal = (idx_price > ma).astype(float)          # 1=牛(持多) 0=熊(空仓)
    signal = signal.shift(1).fillna(0)               # 用上一期信号定本期持仓，防前视

    # 换仓成本：信号翻转时扣双边成本
    turnover = signal.diff().abs().fillna(signal)
    cost = turnover * 2 * cost_side

    strat_ret = signal * idx_ret - cost              # 牛吃收益、熊空仓，扣成本
    bh_ret = idx_ret                                 # buy & hold 一直持有

    strat_nav = np.exp(strat_ret.cumsum())
    bh_nav = np.exp(bh_ret.cumsum())

    def stats(nav: pd.Series, ret: pd.Series) -> dict:
        years = len(ret) / (365 * 24)                # 1h 数据，年化
        total = nav.iloc[-1] - 1
        cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
        dd = (nav / nav.cummax() - 1).min()
        # 夏普（年化，无风险=0）
        sharpe = ret.mean() / (ret.std() + 1e-12) * np.sqrt(365 * 24)
        # 胜率（按「持仓期」算：只在做多时看对不对）
        return {"总收益": total, "年化": cagr, "最大回撤": dd, "夏普": sharpe, "样本小时": len(ret)}

    return {
        "策略": stats(strat_nav, strat_ret),
        "buy_hold": stats(bh_nav, bh_ret),
        "换仓次数": int(turnover.sum()),
        "牛市持仓占比": float((signal == 1).mean()),
        "指数年化": bh_nav.iloc[-1] ** (1 / (len(idx_ret) / (365 * 24))) - 1,
    }


if __name__ == "__main__":
    for mh in (720, 1440, 4320):   # 30日 / 60日 / 180日
        try:
            r = backtest(mh)
        except Exception as exc:
            print(f"{mh//24}日均线 回测失败: {exc}")
            continue
        s, b = r["策略"], r["buy_hold"]
        print(f"\n===== {mh//24} 日均线趋势跟随（{s['样本小时']} 小时，≈{s['样本小时']//8760:.1f} 年）=====")
        print(f"策略  : 总收益 {s['总收益']:+.1%} | 年化 {s['年化']:+.1%} | 最大回撤 {s['最大回撤']:+.1%} | 夏普 {s['夏普']:.2f}")
        print(f"持有  : 总收益 {b['总收益']:+.1%} | 年化 {b['年化']:+.1%} | 最大回撤 {b['最大回撤']:+.1%} | 夏普 {b['夏普']:.2f}")
        print(f"换仓 {r['换仓次数']} 次 | 做多时间占比 {r['牛市持仓占比']:.0%}")
