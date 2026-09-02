# -*- coding: utf-8 -*-
"""组合（插针 + 慢动量 v2）稳健性验证。

本脚本对比：
1. 等权插针 vs 平方加权插针 两种日收益序列（同 1m 数据口径）。
2. w 敏感性：插针权重 w ∈ 0.3~0.9，观察 Sharpe / 回撤变化。
3. 分年：固定 w 下 2024 / 2025 / 2026 逐年 Sharpe / 收益。

用法：python -m backtest.combo_robust
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_ROWS = 1_000_000
LOOKBACK = 15
HOLD_MIN = 720
COST_SIDE = 0.0006
POS_SIZE = 0.05
TH = -0.05
V2_SRC = "data/model_out/replay_all.json"


def load_close():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, open_time, close FROM klines_1m", conn)
    conn.close()
    counts = df.groupby("symbol").size()
    symbols = counts[counts >= MIN_ROWS].index.tolist()
    df = df[df.symbol.isin(symbols)]
    close = df.pivot_table(index="open_time", columns="symbol", values="close",
                           aggfunc="last").sort_index()[symbols]
    return close.values.astype(np.float64), close.index.values.astype(np.int64), symbols


def detect(C, ts, ret, th, K, H):
    T, N = C.shape
    mask = ret <= th
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] = mask[1:] & ~mask[:-1]
    edge[0] = mask[0]
    rows, cols = np.where(edge)
    last = np.full(N, -10**9, dtype=np.int64)
    events = []
    for i, j in zip(rows, cols):
        if i >= T - K - H:
            continue
        if i - last[j] >= H:
            last[j] = i
            entry = C[i + K, j]
            exit_ = C[i + K + H, j]
            if entry > 0 and exit_ > 0:
                events.append((int(ts[i + K]), float(-ret[i, j]),
                               float(exit_ / entry - 1.0 - 2 * COST_SIDE)))
    return events


def daily_rets(events, all_days, wfn):
    """返回 {day_idx: 日收益}，覆盖所有连续交易日（插针空仓日=0）。
    与 combine_track.py 口径一致：组合是「每天持有 v2 + 插针事件日叠加插针」，
    插针空仓的日子 v2 依然在跑，不能把空仓日从组合里剔除。"""
    gross = sum(wfn(d) for _, d, _ in events)
    pos_scale = POS_SIZE * len(events) / gross  # 总资金归一化到等权
    by_day = {}
    for t0, d, r in events:
        by_day[t0 // 86400] = by_day.get(t0 // 86400, 0.0) + pos_scale * wfn(d) * r
    return {d: by_day.get(d, 0.0) for d in all_days}


def v2_daily_rets():
    import json
    from pathlib import Path
    v2_eq = {int(int(x[0]) // 86400): float(x[1])
             for x in json.loads(Path(V2_SRC).read_text(encoding="utf-8"))["legs"]["v2"]["equity"]}
    days = sorted(v2_eq)
    return {days[i]: v2_eq[days[i]] / v2_eq[days[i - 1]] - 1.0 for i in range(1, len(days))}


def sharpe_of(drets):
    a = np.array(drets)
    if a.std() < 1e-12:
        return 0.0, 0.0, 0.0
    eq = np.cumprod(1 + a)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    sharpe = float(a.mean() / a.std() * np.sqrt(365))
    total = float(eq[-1] - 1)
    return sharpe, dd, total


def yearly(days, drets):
    import datetime as dt
    out = {}
    for d, r in zip(days, drets):
        y = dt.datetime.fromtimestamp(d * 86400, tz=dt.timezone.utc).year
        out.setdefault(y, []).append(r)
    return out


def emit_track(pin_daily, v2, w, name, dest):
    """落盘组合轨道 JSON（格式对齐 combine_track.py，供看板 status.py 展示）。"""
    import json
    import datetime as dt
    from pathlib import Path
    common = sorted(set(pin_daily) & set(v2))
    combo = [w * pin_daily[d] + (1 - w) * v2[d] for d in common]
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
    print(f"[emit] {dest}  {name}  Sharpe {sharpe:.2f}  回撤 {dd*100:.1f}%  收益 {stats['total_ret']*100:+.1f}%")


def main():
    C, ts, symbols = load_close()
    T = C.shape[0]
    all_days = list(range(int(ts[0]) // 86400, int(ts[-1]) // 86400 + 1))
    ret = C[LOOKBACK:] / C[:-LOOKBACK] - 1.0
    events = detect(C, ts, ret, TH, LOOKBACK, HOLD_MIN)
    print(f"{len(symbols)} 币 / {TH:.0%} 插针 / 持有 {HOLD_MIN // 60}h / 事件 {len(events)} 笔\n")

    v2 = v2_daily_rets()

    schemes = {
        "等权插针": lambda d: 1.0,
        "平方加权插针": lambda d: np.clip((d / 0.05) ** 2, 0.0, 6.0),
    }

    for name, wfn in schemes.items():
        pin = daily_rets(events, all_days, wfn)
        common = sorted(set(pin) & set(v2))
        pin_a = np.array([pin[d] for d in common])
        v2_a = np.array([v2[d] for d in common])
        print(f"\n===== {name} 组合（共同交易日 {len(common)}）=====")
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

        # 分年（用最优 w）
        w = best[0]
        combo = w * pin_a + (1 - w) * v2_a
        yb = yearly(common, combo)
        print(f"  分年（w={w:.2f}）：")
        for y in sorted(yb):
            a = np.array(yb[y])
            sh, dd, tot = sharpe_of(a)
            print(f"    {y}  n={len(a):>3}  Sharpe {sh:>6.2f}  收益 {tot*100:>+7.1f}%  回撤 {dd*100:>6.1f}%")
        # 各腿单跑对比
        psh, pdd, ptot = sharpe_of(pin_a)
        vsh, vdd, vtot = sharpe_of(v2_a)
        print(f"  单腿对比：插针 Sharpe {psh:.2f}/回撤 {pdd*100:.1f}% | v2 Sharpe {vsh:.2f}/回撤 {vdd*100:.1f}%")
        # 落盘组合轨道（w=0.70 固定，与看板原口径一致）
        if name == "平方加权插针":
            emit_track(pin, v2, 0.70, f"★插针(平方加权)+慢动量组合·70/30",
                       "data/model_out/tracks/combo_pin_v2.json")


if __name__ == "__main__":
    main()
