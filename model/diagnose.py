# -*- coding: utf-8 -*-
"""信号解剖：把「IC 正但 Sharpe 负」拆开定位——毛 vs 净、尾巴(|z|>=1) vs 主体、多 vs 空。

不猜。一次跑完回答三个问题：
1. 毛收益(不扣成本)是正是负？正 → 成本杀人；负 → 信号本身反了。
2. |z|>=1 的尾巴 IC 是正是负？负 → 最自信的预测系统性错(尾巴均值回归)。
3. 多空两边谁在亏？不对称 → 一边的极端币是毒药。

用法: python -m model.diagnose
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, features, train


def _sr(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=0)
    return float(r.mean() / sd) if sd > 0 else 0.0


def _ic(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 4:
        return float("nan")
    v = a.corr(b, method="spearman")
    return float(v) if pd.notna(v) else float("nan")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print(f"[diag] pool {len(pool)} 币，构建面板…")
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    feats = train.feature_cols(panel)
    h = 24
    label = f"fwd_ret_{h}h_cs"
    ret = f"fwd_ret_{h}h"

    subs = []
    for i, (s, e_tr, e_te) in enumerate(train._make_folds(panel)):
        purge = pd.Timedelta(hours=h)
        tr_mask = (panel["open_time"] >= s) & (panel["open_time"] < e_tr - purge)
        te_mask = (panel["open_time"] >= e_tr) & (panel["open_time"] < e_te)
        r = train._fold(panel, tr_mask, te_mask, feats, i, label, ret, h, 1.0)
        if r:
            subs.append(r["sub"])
    all_sub = pd.concat(subs, ignore_index=True)
    tail = all_sub[all_sub["sig"] != 0]
    bulk = all_sub[all_sub["sig"] == 0]
    longs = tail[tail["sig"] == 1]
    shorts = tail[tail["sig"] == -1]

    steps_per_year = 365 * 24 / h

    print("\n=== 信号解剖（24h 截面 z, z=1.0） ===")
    print(f"总样本 {len(all_sub):,} | 下单(|z|>=1) {len(tail):,} | 空仓 {len(bulk):,}")

    # 1. 毛 vs 净（整体组合，按时间步聚合）
    port_net = all_sub.groupby("open_time")["net"].mean()
    port_gross = all_sub.groupby("open_time")["gross"].mean()
    print(f"\n[1] 毛 vs 净 (年化 Sharpe)")
    print(f"    毛 Sharpe = {_sr(port_gross.to_numpy()) * np.sqrt(steps_per_year):+.2f}")
    print(f"    净 Sharpe = {_sr(port_net.to_numpy()) * np.sqrt(steps_per_year):+.2f}")

    # 2. 尾巴 vs 主体 IC
    print(f"\n[2] IC 分解 (pred vs label, Spearman)")
    print(f"    全体 IC   = {_ic(all_sub['pred'], all_sub[label]):+.4f}")
    print(f"    尾巴 IC   = {_ic(tail['pred'], tail[label]):+.4f}  (n={len(tail)})")
    print(f"    主体 IC   = {_ic(bulk['pred'], bulk[label]):+.4f}  (n={len(bulk)})")

    # 3. 多空两边毛收益（每边单独，钱的口径）
    if len(longs) and len(shorts):
        long_gross = longs.groupby("open_time")["gross"].mean()
        short_gross = shorts.groupby("open_time")["gross"].mean()
        print(f"\n[3] 多空毛收益 (年化 Sharpe, 钱口径 ret)")
        print(f"    多头毛利 Sharpe = {_sr(long_gross.to_numpy()) * np.sqrt(steps_per_year):+.2f}  (n={len(longs)})")
        print(f"    空头毛利 Sharpe = {_sr(short_gross.to_numpy()) * np.sqrt(steps_per_year):+.2f}  (n={len(shorts)})")
        print(f"    多头平均 24h 毛收益 = {longs['gross'].mean():+.3%}")
        print(f"    空头平均 24h 毛收益 = {shorts['gross'].mean():+.3%}")

    # 4. 尾巴换手率（成本拖累量级）
    turn = tail.groupby("open_time")["turnover"].mean().mean()
    print(f"\n[4] 成本拖累")
    print(f"    平均单步换手 = {turn:.2f} -> 双边成本 ≈ {turn * config.COST_SIDE * 2:.3%}/步")

    # 5. 尾巴极端币的回看：下单时它们的 24h 已涨跌幅（验证「追涨杀跌被回归」假说）
    if "ret_24h" in tail.columns:
        long_ret = tail[tail["sig"] == 1]["ret_24h"].mean()
        short_ret = tail[tail["sig"] == -1]["ret_24h"].mean()
        print(f"\n[5] 下单时的 24h 已涨跌（验均值回归假说）")
        print(f"    多头标的平均 24h 已涨 {long_ret:+.2%}（追高？）")
        print(f"    空头标的平均 24h 已涨 {short_ret:+.2%}（杀跌？）")


if __name__ == "__main__":
    main()
