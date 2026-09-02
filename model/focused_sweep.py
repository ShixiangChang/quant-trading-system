# -*- coding: utf-8 -*-
"""聚焦地形图：49 池 × 3 目标 × 关键周期，正确口径（train.evaluate 已 expm1 记账）。

目的：sweep_results.csv 是 log-记账 bug 时代的旧数据（有 Sharpe 12 / 回撤 -100% 的爆炸行），
污染了 DSR 的选择偏差基准。这里用正确口径重扫「周期地形图」，回答一个问题：
    有没有比 96h 截面 z（净 Sharpe +1.39）更强的配置？

用法（在项目根目录下）:
    python -m model.focused_sweep            # 12 配置（4 周期 × 3 目标）
    python -m model.focused_sweep --seeds 2  # 提速：临时降 seed 数
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config, features, train


def _market_neutral_labels(panel: pd.DataFrame) -> pd.DataFrame:
    for h in config.HORIZONS:
        col = f"fwd_ret_{h}h"
        panel[f"{col}_mn"] = panel[col] - panel.groupby("open_time")[col].transform("mean")
    return panel


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=None, help="临时覆盖 ENSEMBLE_SEEDS（提速）")
    p.add_argument("--horizons", default="48,72,96,168", help="逗号分隔的周期(h)")
    a = p.parse_args()

    if a.seeds is not None:
        config.ENSEMBLE_SEEDS = a.seeds

    horizons = [int(x) for x in a.horizons.split(",")]

    pool_file = config.OUTPUT_DIR / "probe_pool.txt"
    pool = pool_file.read_text(encoding="utf-8").split()
    print(f"[focus] 49 池面板（{len(pool)} 币，365d 缓存）…", flush=True)
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    panel = _market_neutral_labels(panel)
    feats = train.feature_cols(panel)
    print(f"[focus] 面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 币 | {len(feats)} 特征", flush=True)

    targets = [("原始收益", ""), ("市场中性", "_mn"), ("截面z", "_cs")]
    rows = []
    for h in horizons:
        for tname, suffix in targets:
            label_col = f"fwd_ret_{h}h{suffix}"
            res = train.evaluate(panel, feats, h, label_col, 1.0, ret_col=f"fwd_ret_{h}h")
            if res is None:
                print(f"[focus] {h:>3}h {tname}: 无有效折", flush=True)
                continue
            m = res["metrics"]
            rows.append({
                "horizon_h": h, "target": tname, "z": 1.0,
                "ic": res["ic"], "sharpe": m["sharpe"],
                "total_ret": m["total_ret"], "max_dd": m["max_dd"],
                "pf": m["pf"], "n_steps": m["n"],
                "long_n": res["long_n"], "short_n": res["short_n"],
            })
            print(f"[focus] {h:>3}h {tname:<4} -> IC={res['ic']:+.4f} "
                  f"净Sharpe={m['sharpe']:+.2f} 收益={m['total_ret']:+.1%} 回撤={m['max_dd']:.1%} 步={m['n']}",
                  flush=True)

    results = pd.DataFrame(rows)
    if results.empty:
        print("[focus] 无有效结果")
        return
    out = config.OUTPUT_DIR / "focused_sweep.csv"
    results.to_csv(out, index=False)
    print(f"\n[focus] 结果已存 {out}", flush=True)
    ranked = results.sort_values("sharpe", ascending=False)
    print(ranked.to_string(index=False, float_format=lambda v: f"{v:+.3f}"), flush=True)


if __name__ == "__main__":
    main()
