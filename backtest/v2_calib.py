# -*- coding: utf-8 -*-
"""v2 慢动量腿口径校准：回测 replay.py（96h 持有）vs 实盘 momentum_leg.py（168h 持有）。

背景：发现回测与实盘的 v2 参数不一致——
- replay.py（组合回测用的 replay_all.json）：MOM_HOLD_H=96h，趋势门控二值开关（z>1 满仓）。
- momentum_leg.py（实盘决策）：MOM_HOLD_H=168h（7天=回看30天下沿），趋势门控连续权重（w=tanh(z-1) 乘进仓位）。

本脚本对比四个变体：
  (96h, 二值)   = replay.py 现状
  (96h, 连续)   = 持有不变，门控改连续
  (168h, 二值)  = 持有对齐实盘，门控不变
  (168h, 连续)  = 完全对齐 momentum_leg.py

均用 expanding std 标定 trend_z（walk-forward 无前视，回测正确口径）。
用法：python -m backtest.v2_calib
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.replay import load_klines

MOM_LOOKBACK_H = 720
MOM_SKIP_H = 24
MOM_TOP_N = 20
MOM_POS_MAX = 0.05
MOM_TREND_OFFSET = 1.0
ATR_MULT = 3.0
TREND_MA_HOURS = 720


def prep(df: pd.DataFrame, hold_h: int) -> pd.DataFrame:
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["logret"] = df.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))
    df["mom_720"] = df.groupby("symbol")["logret"].transform(
        lambda s: s.rolling(MOM_LOOKBACK_H).sum().shift(MOM_SKIP_H))
    tr = df["high"] - df["low"]
    df["atr"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14).mean())
    df["atr_norm"] = df["atr"] / df["close"]

    # 未来 hold 期间 low/close（严格 t 之后），参数化持有期
    df = df.iloc[::-1]
    df["f_low"] = df.groupby("symbol")["low"].transform(
        lambda s: s.shift(1).rolling(hold_h, min_periods=1).min())
    df["f_close"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(hold_h))
    df = df.iloc[::-1]

    # 全池等权几何指数 + 趋势 z（expanding std，无前视）
    idx_ret = df.groupby("open_time")["logret"].mean()
    idx = np.exp(idx_ret.cumsum())
    ma = idx.rolling(TREND_MA_HOURS).mean()
    dev = idx / ma - 1.0
    trend_z = dev / dev.expanding(min_periods=TREND_MA_HOURS).std()
    df["trend_z"] = df["open_time"].map(trend_z.to_dict())
    return df


def replay(df: pd.DataFrame, continuous: bool) -> dict:
    d = df.dropna(subset=["mom_720", "atr_norm", "trend_z"]).copy()
    cross = d[d["open_time"] % 86400 == 0].sort_values("open_time")
    times = sorted(cross["open_time"].unique())
    cross = cross[cross["open_time"].isin(times)]

    nav = 1.0
    eq = []
    rets = []
    n_pos_days = 0
    for t in times:
        snap = cross[cross["open_time"] == t]
        z = float(snap["trend_z"].iloc[0])
        w = max(0.0, np.tanh(z - MOM_TREND_OFFSET))
        if w <= 1e-6:
            rets.append(0.0)
            continue
        n_pos_days += 1
        top = snap.nlargest(MOM_TOP_N, "mom_720")
        leg_ret = 0.0
        for _, r in top.iterrows():
            entry = float(r["close"])
            atr = max(float(r["atr_norm"]) or 0.03, 0.005)
            stop = entry * (1.0 - ATR_MULT * atr)
            if pd.isna(r["f_close"]):
                continue
            exit_px = stop if (pd.notna(r["f_low"]) and r["f_low"] <= stop) else float(r["f_close"])
            one = exit_px / entry - 1.0
            leg_ret += one * MOM_POS_MAX * (w if continuous else 1.0)
        nav *= (1.0 + leg_ret)
        rets.append(leg_ret)
        eq.append([t, nav])

    a = np.array(rets)
    sharpe = float(a.mean() / (a.std() + 1e-12) * np.sqrt(365))
    peak = np.maximum.accumulate(np.array([e[1] for e in eq]))
    dd = float((np.array([e[1] for e in eq]) - peak).min() / 1.0) if eq else 0.0
    # 正确算回撤
    navs = np.array([e[1] for e in eq]) if eq else np.array([1.0])
    peak = np.maximum.accumulate(navs)
    dd = float(((navs - peak) / peak).min())
    total = float(nav - 1)
    return {"sharpe": sharpe, "dd": dd, "total": total, "n_pos_days": n_pos_days,
            "annual": float(nav ** (365 / len(rets)) - 1) if rets else 0.0}


def main():
    print("加载 1h K线...")
    df = load_klines()

    print(f"{'持有期':>6} {'门控':>4} {'Sharpe':>7} {'回撤':>8} {'收益':>8} {'年化':>8} {'持仓天数':>6}")
    for hold_h in [96, 168]:
        d = prep(df, hold_h)
        for continuous in [False, True]:
            r = replay(d, continuous)
            gate = "连续" if continuous else "二值"
            print(f"{hold_h:>6}h {gate:>4} {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}% "
                  f"{r['total']*100:>+7.1f}% {r['annual']*100:>+7.1f}% {r['n_pos_days']:>6}")


if __name__ == "__main__":
    main()
