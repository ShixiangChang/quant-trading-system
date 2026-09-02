# -*- coding: utf-8 -*-
"""精确 DSR 复核：96h 截面 z 信号的收益序列 + 正确口径试验集（focused_sweep 12 配置）。

回答：修正了 sweep_results.csv 的 log-记账 bug 后，DSR 到底过不过？卡在哪（Sharpe 不够 vs 肥尾）？
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from . import config, features, train
from .gate import _sr, probabilistic_sharpe_ratio, deflated_sharpe_ratio, min_track_record_length


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print("[dsr] 面板…", flush=True)
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    feats = train.feature_cols(panel)

    res = train.evaluate(panel, feats, 96, "fwd_ret_96h_cs", 1.0, ret_col="fwd_ret_96h")
    m = res["metrics"]
    r = res["port"]["port_ret"].to_numpy()
    print(f"[dsr] 96h截面z: 净Sharpe={_sr(r):+.3f} 收益={m['total_ret']:+.1%} 回撤={m['max_dd']:.1%} 步={m['n']}",
          flush=True)

    # 肥尾诊断
    g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False))
    sr = _sr(r); T = len(r)
    var_sr = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    print(f"[dsr] 偏度={g3:+.2f} 峰度(原始)={g4:.1f} (正态=3) → var_sr={var_sr:.1f}", flush=True)
    print(f"[dsr] 说明：var_sr 越大，Sharpe 估计越不可靠（肥尾导致）；正态 var_sr≈1", flush=True)

    # 正确口径试验集：focused_sweep 12 配置的 sharpe
    fp = config.OUTPUT_DIR / "focused_sweep.csv"
    trials = pd.read_csv(fp)["sharpe"].astype(float).tolist()
    print(f"[dsr] 试验集 N={len(trials)} std={np.std(trials, ddof=1):.3f}", flush=True)

    psr0 = probabilistic_sharpe_ratio(r, 0.0)
    dsr = deflated_sharpe_ratio(r, trials)
    mtrl = min_track_record_length(r)
    print(f"\n=== 96h 截面 z 精确复核 ===", flush=True)
    print(f"  PSR(vs 0)   = {psr0:.3f}", flush=True)
    print(f"  DSR(12试验) = {dsr:.3f}  {'✅ 过' if dsr >= 0.95 else '❗ 未过'}", flush=True)
    print(f"  MinTRL      = {mtrl:.0f} 步（当前 {T} 步）", flush=True)


if __name__ == "__main__":
    main()
