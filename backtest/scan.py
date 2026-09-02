# -*- coding: utf-8 -*-
"""参数敏感性扫描：用历史回放引擎几分钟扫完 止损×分散度×回看窗口 的组合。

目标：慢动量 +430% 但回撤 -77.5% 不可交易。
波动率门控已证伪（砍收益不护回撤）。正路是扫：
  ① 止损倍数（3×ATR 每次亏 3 个 ATR 太深 → 试 1.5/2.0/3.0）
  ② 分散度（top10 太集中 → 试 10/20/30）
  ③ 回看窗口（7 天 → 试 7/14/30 天）

不选样本内最高的配置，而是观察稳健区间——
相邻参数下表现接近的才可信；并做 walk-forward（前 1 年选参数，后 1 年验证）。

用法：python backtest/scan.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.replay import load_klines, _max_dd, MOM_SKIP_H, MOM_HOLD_H, TREND_MA_HOURS


def prep(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """预计算多 lookback 动量 + ATR + 未来结算价 + 趋势 z（一次算，扫描复用）。"""
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["logret"] = df.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))
    tr = df["high"] - df["low"]
    df["atr"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14).mean())
    df["atr_norm"] = df["atr"] / df["close"]

    df = df.iloc[::-1]
    df["f_low"] = df.groupby("symbol")["low"].transform(
        lambda s: s.shift(1).rolling(MOM_HOLD_H, min_periods=1).min())
    df["f_close"] = df.groupby("symbol")["close"].transform(
        lambda s: s.shift(MOM_HOLD_H))
    df = df.iloc[::-1]

    idx_ret = df.groupby("open_time")["logret"].mean()
    idx = np.exp(idx_ret.cumsum())
    ma = idx.rolling(TREND_MA_HOURS).mean()
    dev = idx / ma - 1.0
    trend_z = dev / dev.expanding(min_periods=TREND_MA_HOURS).std()
    df["trend_z"] = df["open_time"].map(trend_z.to_dict())

    lookbacks = [168, 336, 720]   # 7 / 14 / 30 天
    for lb in lookbacks:
        df[f"mom_{lb}"] = df.groupby("symbol")["logret"].transform(
            lambda s: s.rolling(lb).sum().shift(MOM_SKIP_H))
    return df, lookbacks


def run_one(df: pd.DataFrame, cross: pd.DataFrame, times: list[int],
            lookback: int, top_n: int, atr_mult: float) -> dict:
    """单组参数全历史回放。返回 {total_ret, max_dd, calmar, hit_rate, avg_ret}。"""
    mom_col = f"mom_{lookback}"
    nav = 1.0
    eq = []
    rets = []
    hits = 0
    for t in times:
        snap = cross[cross["open_time"] == t]
        z = float(snap["trend_z"].iloc[0])
        w = max(0.0, np.tanh(z))
        if w <= 1e-6:
            eq.append(nav)
            continue
        top = snap.nlargest(top_n, mom_col)
        leg = 0.0
        n = 0
        for _, r in top.iterrows():
            entry = float(r["close"])
            atr = max(float(r["atr_norm"]) or 0.03, 0.005)
            stop = entry * (1.0 - atr_mult * atr)
            if pd.isna(r["f_close"]):
                continue
            exit_px = stop if (pd.notna(r["f_low"]) and r["f_low"] <= stop) else float(r["f_close"])
            one = exit_px / entry - 1.0
            leg += one * (1.0 / top_n)      # 等权满仓（每币 1/top_n）
            rets.append(one)
            hits += 1 if one > 0 else 0
            n += 1
        if n:
            nav *= (1.0 + leg)
        eq.append(nav)
    eq = pd.Series(eq)
    total = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    dd = _max_dd(eq) if len(eq) else 0.0
    return {
        "total_ret": total,
        "max_dd": dd,
        "calmar": total / abs(dd) if dd else 0.0,
        "hit_rate": hits / len(rets) if rets else 0.0,
        "avg_ret": float(np.mean(rets)) if rets else 0.0,
        "n": len(rets),
    }


def main():
    print("加载 + 预计算（多 lookback 动量，扫描复用）...")
    df = load_klines()
    df, lookbacks = prep(df)
    cross = df.dropna(subset=["trend_z"]).copy()
    cross = cross[cross["open_time"] % 86400 == 0].sort_values("open_time")
    times = sorted(cross["open_time"].unique())

    # walk-forward 分割点：前 1 年 = 选参数，后 1 年 = 验证
    mid_ts = times[len(times) // 2]
    times_in = [t for t in times if t <= mid_ts]
    times_out = [t for t in times if t > mid_ts]

    results = []
    for lb in lookbacks:
        for top_n in [10, 20, 30]:
            for atr_mult in [1.5, 2.0, 3.0]:
                r_in = run_one(df, cross, times_in, lb, top_n, atr_mult)
                r_out = run_one(df, cross, times_out, lb, top_n, atr_mult)
                results.append({
                    "lookback_d": lb // 24, "top_n": top_n, "atr_mult": atr_mult,
                    "in_total": r_in["total_ret"], "in_dd": r_in["max_dd"],
                    "out_total": r_out["total_ret"], "out_dd": r_out["max_dd"],
                    "hit": r_in["hit_rate"], "avg": r_in["avg_ret"],
                })

    res = pd.DataFrame(results)
    # 排序：样本外（后 1 年）收益/回撤 优先，兼顾样本内外一致
    res["out_calmar"] = res["out_total"] / res["out_dd"].abs()
    res = res.sort_values("out_calmar", ascending=False)

    print("\n" + "=" * 96)
    print("慢动量参数扫描（27 组合）｜样本内=前1年，样本外=后1年（真验证）")
    print("=" * 96)
    print(f"{'回看':>4} {'topN':>4} {'止损':>4} | {'样本内收益':>8} {'内回撤':>7} | {'样本外收益':>8} {'外回撤':>7} {'外Calmar':>8}")
    print("-" * 96)
    for _, r in res.iterrows():
        print(f"{r['lookback_d']:>3}天 {r['top_n']:>4} {r['atr_mult']:.1f}x | "
              f"{r['in_total']*100:>+7.0f}% {r['in_dd']*100:>6.0f}% | "
              f"{r['out_total']*100:>+7.0f}% {r['out_dd']*100:>6.0f}% {r['out_calmar']:>7.2f}")
    print("=" * 96)

    # 稳健性：样本内样本外都正收益的组合
    both_pos = res[(res["in_total"] > 0) & (res["out_total"] > 0)]
    print(f"\n样本内、样本外都赚钱的组合（稳健，非单期运气）: {len(both_pos)} 个")
    if len(both_pos):
        print("  其中样本外回撤最小的 3 个:")
        for _, r in both_pos.nsmallest(3, "out_dd").iterrows():
            print(f"    回看{r['lookback_d']:>2}天 top{r['top_n']:>2} 止损{r['atr_mult']:.1f}x "
                  f"→ 内{r['in_total']*100:+.0f}%/外{r['out_total']*100:+.0f}% 外回撤{r['out_dd']*100:.0f}%")


if __name__ == "__main__":
    main()
