# -*- coding: utf-8 -*-
"""验证纪律：成本压力 + 参数敏感性 + 币种一致性 三关自动体检。

在 gate.py（时间 OOS + DSR/PSR/MinTRL）之上再补三关，让「候选信号」自动过体检、
自动检查，而不是手动逐个查：

1. 成本压力：COST_SIDE ×1/2/3，看净 Sharpe 是否还 >0 —— 信号够不够强到扛真实成本
2. 参数敏感性：z 阈值 0.8/1.0/1.2 + horizon 邻近档扰动，看信号方向是否还稳
3. 币种一致性：pool 随机分 A/B 两半，看两边是否同号 —— 信号是否过拟合到少数币

口径：截面 z 标签（fwd_ret_*h_cs）训练、原始收益（fwd_ret_*h）记账，对齐 gate.py / sweep。

用法: python -m model.gate_hard [--horizon 96] [--fast]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config, features, train


def _run(panel: pd.DataFrame, feats: list[str], h: int, z: float):
    label = f"fwd_ret_{h}h_cs"
    ret = f"fwd_ret_{h}h"
    return train.evaluate(panel, feats, h, label, z, ret_col=ret)


# ------------------------------------------------ 1. 成本压力
def cost_stress(panel, feats, h, z) -> list[tuple]:
    base = config.COST_SIDE
    print("=== 1. 成本压力测试（成本 ×1/2/3，看信号扛不扛得住） ===")
    print(f"{'单边成本':<12} {'净Sharpe':>9} {'收益':>8} {'回撤':>8}")
    rows = []
    for mult in (1.0, 2.0, 3.0):
        config.COST_SIDE = base * mult
        try:
            res = _run(panel, feats, h, z)
        finally:
            config.COST_SIDE = base
        if res is None:
            print(f"  {base * mult:.4%}{'':<6} 无有效折")
            continue
        m = res["metrics"]
        rows.append((mult, m["sharpe"], m["total_ret"], m["max_dd"]))
        print(f"  {base * mult:.4%}{'':<6} {m['sharpe']:>+9.2f} {m['total_ret']:>+8.1%} {m['max_dd']:>8.1%}")
    if rows:
        worst = rows[-1][1]  # 3x 成本下的 Sharpe
        print("  → " + ("✅ 通过：3x 成本下仍正 Sharpe（信号有成本安全垫）"
                        if worst > 0 else "❌ 未通过：成本压力下信号转负（alpha 太薄）"))
    return rows


# ------------------------------------------------ 2. 参数敏感性
def param_sensitivity(panel, feats, h, z) -> None:
    print("\n=== 2. 参数敏感性（扰动参数观察信号方向是否稳定） ===")
    print(f"{'z 阈值':<10} {'Sharpe':>9} {'IC':>9}")
    sharpes_z = []
    for zz in (0.8, 1.0, 1.2):
        res = _run(panel, feats, h, zz)
        if res:
            sharpes_z.append(res["metrics"]["sharpe"])
            print(f"  {zz:<10} {res['metrics']['sharpe']:>+9.2f} {res['ic']:>+9.4f}")

    neighbors = [x for x in sorted(config.HORIZONS) if h // 2 <= x <= h * 2]
    print(f"{'horizon':<10} {'Sharpe':>9} {'IC':>9}")
    sharpes_h = []
    for hh in neighbors:
        if hh == h:
            continue
        res = _run(panel, feats, hh, z)
        if res:
            sharpes_h.append(res["metrics"]["sharpe"])
            print(f"  {hh:<10} {res['metrics']['sharpe']:>+9.2f} {res['ic']:>+9.4f}")

    if sharpes_z and sharpes_h:
        z_stable = all(s > 0 for s in sharpes_z) or all(s < 0 for s in sharpes_z)
        h_stable = all(s > 0 for s in sharpes_h) or all(s < 0 for s in sharpes_h)
        print(f"  → z 扰动{'同号（稳）' if z_stable else '异号（脆）'}；"
              f"horizon 扰动{'同号（稳）' if h_stable else '异号（脆）'}")


# ------------------------------------------------ 3. 币种一致性
def coin_split(panel, feats, h, z, seed: int = 42):
    print("\n=== 3. 币种一致性（随机分 A/B 两半，看是否同号） ===")
    syms = np.array(panel["symbol"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(syms)
    half = len(syms) // 2
    ra = _run(panel[panel["symbol"].isin(syms[:half])], feats, h, z)
    rb = _run(panel[panel["symbol"].isin(syms[half:])], feats, h, z)
    if ra and rb:
        sa, sb = ra["metrics"]["sharpe"], rb["metrics"]["sharpe"]
        print(f"  A组({half}币) Sharpe={sa:+.2f} IC={ra['ic']:+.4f}")
        print(f"  B组({len(syms) - half}币) Sharpe={sb:+.2f} IC={rb['ic']:+.4f}")
        same = (sa > 0) == (sb > 0)
        print("  → " + ("✅ 一致：两边同号，信号非单边币过拟合"
                        if same else "❌ 不一致：信号过拟合到少数币"))
    return ra, rb


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="验证纪律：成本/参数/币种三关")
    p.add_argument("--horizon", type=int, default=96, help="持仓期(h)，默认 96（回测最优点）")
    p.add_argument("--fast", action="store_true", help="1-seed 快检（省时，方向参考用）")
    a = p.parse_args()

    if a.fast:
        config.ENSEMBLE_SEEDS = 1

    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print(f"[gate_hard] 构建面板（{len(pool)} 币，{a.horizon}h 截面 z，"
          f"{'1-seed 快检' if a.fast else str(config.ENSEMBLE_SEEDS) + '-seed'}）…")
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    feats = train.feature_cols(panel)
    print(f"[gate_hard] 面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 币 | 特征 {len(feats)} 个\n")

    cost_stress(panel, feats, a.horizon, config.SIGNAL_Z)
    param_sensitivity(panel, feats, a.horizon, config.SIGNAL_Z)
    coin_split(panel, feats, a.horizon, config.SIGNAL_Z)


if __name__ == "__main__":
    main()
