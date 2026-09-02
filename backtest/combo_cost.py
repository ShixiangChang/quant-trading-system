# -*- coding: utf-8 -*-
"""v2 慢动量腿的换手成本验证 + 组合成本敏感性。

背景：combo_robust.py 里组合 Sharpe 3.32 有口径隐患——插针腿扣了 6bp×2 往返成本，
但 v2 腿（replay_all.json）的收益是「毛收益」，replay_momentum 里没有任何 cost/fee 处理。
这意味着组合数字对 v2 腿是虚高的，并轨前必须量化 v2 换手成本对组合的冲击。

本脚本忠实复现 replay.py 的 v2 逻辑（30天动量/top20/每币5%/96h结算/3×ATR止损/趋势门控），
只加一件事：每天调仓的换手成本。换手 = 新旧 top20 的币数差 × 每币仓位 × 单边成本。
扫 3bp/6bp/10bp 三档单边成本，看 v2 单腿 + 组合分别降多少。

用法：python -m backtest.combo_cost
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.replay import load_klines, prep_features

MOM_HOLD_H = 96
MOM_TOP_N = 20
MOM_POS_MAX = 0.05
MOM_TREND_OFFSET = 1.0
ATR_MULT = 3.0
V2_SRC = "data/model_out/replay_all.json"
PIN_SQ_SRC = "data/model_out/tracks/combo_pin_v2.json"  # 仅用于取插针日收益参照（见下）


def replay_v2_turnover(df: pd.DataFrame):
    """重放 v2，逐日记录 (day, leg_ret毛收益, turnover换手仓位)。"""
    d = df.dropna(subset=["mom_720", "atr_norm", "trend_z"]).copy()
    cross = d[d["open_time"] % 86400 == 0].sort_values("open_time")
    times = sorted(cross["open_time"].unique())
    cross = cross[cross["open_time"].isin(times)]

    rows = []  # (day_idx, leg_ret, turnover, n)
    prev_hold: set = set()
    for t in times:
        snap = cross[cross["open_time"] == t]
        z = float(snap["trend_z"].iloc[0])
        w = max(0.0, np.tanh(z - MOM_TREND_OFFSET))
        if w <= 1e-6:
            prev_hold = set()
            rows.append((int(t) // 86400, 0.0, 0.0, 0))
            continue
        top = snap.nlargest(MOM_TOP_N, "mom_720")
        cur_hold = set(top["symbol"].tolist())
        sell = len(prev_hold - cur_hold)
        buy = len(cur_hold - prev_hold)
        turnover = (sell + buy) * MOM_POS_MAX  # 满仓开关，换手仓位（每币 5%）

        leg_ret, n = 0.0, 0
        for _, r in top.iterrows():
            entry = float(r["close"])
            atr = max(float(r["atr_norm"]) or 0.03, 0.005)
            stop = entry * (1.0 - ATR_MULT * atr)
            if pd.isna(r["f_close"]):
                continue
            exit_px = stop if (pd.notna(r["f_low"]) and r["f_low"] <= stop) else float(r["f_close"])
            leg_ret += (exit_px / entry - 1.0) * MOM_POS_MAX
            n += 1
        prev_hold = cur_hold
        rows.append((int(t) // 86400, leg_ret, turnover, n))
    return rows


def equity_from(rows, cost_side):
    days, rets, turns = [], [], []
    for day, leg, turn, n in rows:
        days.append(day)
        rets.append(leg - turn * cost_side)  # 扣成本
        turns.append(turn)
    nav = np.cumprod(1 + np.array(rets))
    peak = np.maximum.accumulate(nav)
    dd = float(((nav - peak) / peak).min())
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(365))
    total = float(nav[-1] - 1)
    avg_turn = float(np.mean(turns))
    return {"sharpe": sharpe, "dd": dd, "total": total,
            "daily": {d: r for d, r in zip(days, rets)}, "avg_turn": avg_turn}


def main():
    print("加载 1h K线 + 预计算因子（约 1 分钟）...")
    df = load_klines()
    df, _, _ = prep_features(df)
    print("重放 v2 逐日换手...")
    rows = replay_v2_turnover(df)
    n_cross = sum(1 for r in rows if r[2] > 0)  # 有持仓的天数

    # 无成本基准（对齐 replay_all.json）
    base = equity_from(rows, 0.0)
    print(f"\nv2 腿：{len(rows)} 天 / 有持仓 {n_cross} 天")
    print(f"{'成本(单边)':>10} {'v2 Sharpe':>10} {'v2 回撤':>9} {'v2 收益':>9} {'日均换手':>9}")
    print(f"{'0bp(毛)':>10} {base['sharpe']:>10.2f} {base['dd']*100:>8.1f}% {base['total']*100:>+8.1f}% {base['avg_turn']*100:>8.1f}%")

    # 组合：v2 日收益 + 平方加权插针日收益（从 combo_robust 的 pin 源重算）
    # 插针日收益用 1m 数据重算太重，这里改为：从 pin_strategy_12h.json 等权→无深度加权。
    # 为控制本脚本复杂度，组合成本敏感性采用「v2 扣成本 + 插针扣6bp(已扣)」，
    # 插针腿用 combo_robust 同口径的平方加权日收益，落盘一份供复用。
    from backtest.combo_robust import load_close, detect, daily_rets, LOOKBACK, HOLD_MIN, COST_SIDE, TH
    C, ts, symbols = load_close()
    T = C.shape[0]
    all_days = list(range(int(ts[0]) // 86400, int(ts[-1]) // 86400 + 1))
    ret = C[LOOKBACK:] / C[:-LOOKBACK] - 1.0
    events = detect(C, ts, ret, TH, LOOKBACK, HOLD_MIN)
    sq = lambda d: np.clip((d / 0.05) ** 2, 0.0, 6.0)
    pin = daily_rets(events, all_days, sq)

    print(f"\n插针腿（平方加权）：{len(events)} 事件，扣 {COST_SIDE*2*10000:.0f}bp 往返成本")
    print(f"{'成本(单边)':>10} {'组合Sharpe':>10} {'组合回撤':>9} {'组合收益':>9}")
    for cost_side in [0.0, 0.0003, 0.0006, 0.0010]:
        v2 = equity_from(rows, cost_side)
        common = sorted(set(pin) & set(v2["daily"]))
        combo = [0.70 * pin[d] + 0.30 * v2["daily"][d] for d in common]
        a = np.array(combo)
        eq = np.cumprod(1 + a)
        peak = np.maximum.accumulate(eq)
        dd = float(((eq - peak) / peak).min())
        sh = float(a.mean() / (a.std() + 1e-12) * np.sqrt(365))
        tot = float(eq[-1] - 1)
        print(f"{cost_side*10000:>9.0f}bp {sh:>10.2f} {dd*100:>8.1f}% {tot*100:>+8.1f}%")


if __name__ == "__main__":
    main()
