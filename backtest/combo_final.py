# -*- coding: utf-8 -*-
"""组合口径重算：平方加权插针 + v2(96h 连续门控)。

背景：回测与实盘的 v2 参数存在差异——
- 回测 replay.py 的 v2 = 96h 二值（Sharpe 2.26 / 回撤 -43.3%）。
- 实盘 momentum_leg.py = 168h 连续（Sharpe 1.95 / 回撤 -50.5%）。
- 满足「回撤 ≤ 40%」且 Sharpe 较高的口径 = 96h 连续（2.15 / -28.5%）。

本脚本用「96h 连续」v2 重算组合，得到口径自洽的组合结果。

用法：python -m backtest.combo_final
"""
from __future__ import annotations

import numpy as np

from backtest.v2_calib import prep, MOM_TOP_N, MOM_POS_MAX, MOM_TREND_OFFSET, ATR_MULT
from backtest.replay import load_klines
from backtest.combo_robust import (load_close, detect, daily_rets,
                                   LOOKBACK, HOLD_MIN, COST_SIDE, TH, sharpe_of)


def v2_daily_96_continuous(df) -> dict:
    """v2 腿（96h 持有 + 连续趋势门控）逐日收益 {day: ret}，空仓天=0。"""
    d = prep(df, 96)
    cross = d[d["open_time"] % 86400 == 0].sort_values("open_time")
    times = sorted(cross["open_time"].unique())
    cross = cross[cross["open_time"].isin(times)]
    out = {}
    for t in times:
        snap = cross[cross["open_time"] == t]
        z = float(snap["trend_z"].iloc[0])
        w = max(0.0, np.tanh(z - MOM_TREND_OFFSET))
        day = int(t) // 86400
        if w <= 1e-6:
            out[day] = 0.0
            continue
        top = snap.nlargest(MOM_TOP_N, "mom_720")
        leg_ret = 0.0
        for _, r in top.iterrows():
            entry = float(r["close"])
            atr = max(float(r["atr_norm"]) or 0.03, 0.005)
            stop = entry * (1.0 - ATR_MULT * atr)
            if pd_isna(r["f_close"]):
                continue
            exit_px = stop if (pd_notna(r["f_low"]) and r["f_low"] <= stop) else float(r["f_close"])
            leg_ret += (exit_px / entry - 1.0) * MOM_POS_MAX * w
        out[day] = leg_ret
    return out


def pd_isna(x):
    import pandas as pd
    return pd.isna(x)


def pd_notna(x):
    import pandas as pd
    return pd.notna(x)


def emit_track(pin_daily, v2_daily, w, name, dest):
    """落盘组合轨道 JSON（96h 连续 v2 + 平方加权插针），格式对齐 combine_track.py。"""
    import json
    import datetime as dt
    from pathlib import Path
    common = sorted(set(pin_daily) & set(v2_daily))
    combo = [w * pin_daily[d] + (1 - w) * v2_daily[d] for d in common]
    eq = np.cumprod(1 + np.array(combo))
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    annual = float(eq[-1] ** (365 / len(combo)) - 1)
    sharpe = float(np.mean(combo) / (np.std(combo) + 1e-12) * np.sqrt(365))
    stats = {
        "total_ret": float(eq[-1] - 1), "max_drawdown": dd,
        "hit_rate": float((np.array(combo) > 0).mean()),
        "annual": annual, "sharpe": sharpe, "events": 0, "odds": 0.0,
    }
    fmt = lambda d: dt.datetime.fromtimestamp(d * 86400, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    out = {
        "name": name, "start": fmt(common[0]), "end": fmt(common[-1]),
        "stats": stats,
        "equity": [[int(d) * 86400, round(float(e), 4)] for d, e in zip(common, eq)],
        "monthly": {},
    }
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[emit] {dest}  {name}  Sharpe {sharpe:.2f}  回撤 {dd*100:.1f}%  "
          f"收益 {stats['total_ret']*100:+.1f}%  年化 {annual*100:+.1f}%")


def main():
    print("加载 1h K线 + 算 v2(96h连续) 日收益...")
    df = load_klines()
    v2 = v2_daily_96_continuous(df)
    v2a = np.array(list(v2.values()))
    print(f"  v2(96h连续)：Sharpe {sharpe_of(v2a)[0]:.2f} / 回撤 {sharpe_of(v2a)[1]*100:.1f}% / 收益 {sharpe_of(v2a)[2]*100:+.1f}%")

    print("加载 1m K线 + 算平方加权插针日收益...")
    C, ts, symbols = load_close()
    T = C.shape[0]
    all_days = list(range(int(ts[0]) // 86400, int(ts[-1]) // 86400 + 1))
    ret = C[LOOKBACK:] / C[:-LOOKBACK] - 1.0
    events = detect(C, ts, ret, TH, LOOKBACK, HOLD_MIN)
    sq = lambda d: np.clip((d / 0.05) ** 2, 0.0, 6.0)
    pin = daily_rets(events, all_days, sq)
    print(f"  插针(平方加权)：{len(events)} 事件")

    common = sorted(set(pin) & set(v2))
    pin_a = np.array([pin[d] for d in common])
    v2_a = np.array([v2[d] for d in common])
    print(f"\n组合（共同交易日 {len(common)}）")
    print(f"{'w插针':>6} {'Sharpe':>7} {'回撤':>8} {'总收益':>8} {'年化':>8}")
    best = None
    for w in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        combo = w * pin_a + (1 - w) * v2_a
        sh, dd, tot = sharpe_of(combo)
        annual = float((1 + tot) ** (365 / len(combo)) - 1)
        print(f"{w:>6.2f} {sh:>7.2f} {dd*100:>7.1f}% {tot*100:>+7.1f}% {annual*100:>+7.1f}%")
        if best is None or sh > best[1]:
            best = (w, sh, dd, tot)
    print(f"→ 最优 w={best[0]:.2f}  Sharpe {best[1]:.2f}  回撤 {best[2]*100:.1f}%  收益 {best[3]*100:+.1f}%")

    # 分年
    import datetime as dt
    w = best[0]
    combo = w * pin_a + (1 - w) * v2_a
    yb = {}
    for d, r in zip(common, combo):
        y = dt.datetime.fromtimestamp(d * 86400, tz=dt.timezone.utc).year
        yb.setdefault(y, []).append(r)
    print(f"  分年（w={w:.2f}）：")
    for y in sorted(yb):
        a = np.array(yb[y])
        sh, dd, tot = sharpe_of(a)
        print(f"    {y}  n={len(a):>3}  Sharpe {sh:>6.2f}  收益 {tot*100:>+7.1f}%  回撤 {dd*100:>6.1f}%")

    # 落盘组合轨道（96h 连续 v2 + 平方加权插针，w=最优）
    emit_track(pin, v2, w, f"★插针(平方加权)+慢动量(96h连续)·{int(w*100)}/{int((1-w)*100)}",
               "data/model_out/tracks/combo_pin_v2.json")


if __name__ == "__main__":
    main()
