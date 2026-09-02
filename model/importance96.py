# -*- coding: utf-8 -*-
"""96h 截面 z 信号（49 池、无微观）的因子重要性：跨折跨 seed 聚合 LightGBM gain。
回答「模型用了哪些因子、权重如何」——给的是 gain 占比，不是线性权重。
"""
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
    print(f"[imp] 构建 49 池面板（无微观结构，对齐回测闸门口径）…", flush=True)
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    feats = train.feature_cols(panel)
    print(f"[imp] 面板 {len(panel):,} 行 | 特征 {len(feats)} 个", flush=True)

    h, label, ret = 96, "fwd_ret_96h_cs", "fwd_ret_96h"
    all_imp = {}
    folds = train._make_folds(panel)
    for i, (s, e_tr, e_te) in enumerate(folds):
        purge = pd.Timedelta(hours=h)
        tr_mask = (panel["open_time"] >= s) & (panel["open_time"] < e_tr - purge)
        te_mask = (panel["open_time"] >= e_tr) & (panel["open_time"] < e_te)
        res = train._fold(panel, tr_mask, te_mask, feats, i, label, ret, h, 1.0)
        if res is None:
            continue
        for k, v in res["importance"].items():
            all_imp[k] = all_imp.get(k, 0.0) + v

    imp = pd.Series(all_imp).sort_values(ascending=False)
    total = imp.sum()
    print(f"\n=== 96h 截面 z 因子重要性（gain，跨 {len(folds)} 折 × {config.ENSEMBLE_SEEDS} seed 聚合）===", flush=True)
    for feat, val in imp.head(30).items():
        print(f"  {feat:<20s} {val:>12,.0f}  {val / total * 100:5.1f}%", flush=True)
    imp.to_csv(config.OUTPUT_DIR / "feature_importance_96h.csv")


if __name__ == "__main__":
    main()
