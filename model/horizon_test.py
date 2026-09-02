# -*- coding: utf-8 -*-
"""持仓期地形图：50 池、截面 z 目标、z=1.0，扫 24h→168h，找「信号 > 成本」的净正点。

背景：24h 净 Sharpe 已确认负（-1.74）。成本拖累随 horizon 线性下降（0.12%→0.03%/天→0.017%/天），
而加密截面动量/费率 edge 文献里在周级。这个脚本回答：有没有一个 horizon 让净 Sharpe 转正。

用法: python -m model.horizon_test
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, features, train


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print(f"[ht] pool {len(pool)} 币，构建面板…")
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    feats = train.feature_cols(panel)
    print(f"[ht] 面板 {len(panel):,} 行 | {len(feats)} 特征")

    print("\n=== 持仓期地形图（50 池 / 截面z / z=1.0 / 净口径） ===")
    print(f"{'h':>4} {'IC':>8} {'毛Sharpe':>9} {'净Sharpe':>9} {'总收益':>8} {'回撤':>8} {'步数':>6}")
    for h in (24, 48, 72, 96, 168):
        label = f"fwd_ret_{h}h_cs"
        ret = f"fwd_ret_{h}h"
        res = train.evaluate(panel, feats, h, label, 1.0, ret_col=ret)
        if res is None:
            print(f"{h:>4}  无有效折")
            continue
        m = res["metrics"]
        mg = res["metrics_gross"]
        print(f"{h:>4} {res['ic']:>+8.4f} {mg['sharpe']:>+9.2f} {m['sharpe']:>+9.2f} "
              f"{m['total_ret']:>+8.1%} {m['max_dd']:>8.1%} {m['n']:>6}")


if __name__ == "__main__":
    main()
