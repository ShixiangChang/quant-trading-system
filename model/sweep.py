# -*- coding: utf-8 -*-
"""快速验证引擎（sweep）：一条命令扫过 horizon × 目标 × 阈值，输出排名榜。

解决"验证闭环慢"：特征只算一遍（K 线已缓存），之后每个配置只是换标签/阈值重训，
把"一次跑一个假设"变成"一条命令跑出地形图"。

用法（在项目根目录下）:
    python -m model.sweep
"""
from __future__ import annotations

import sys

import pandas as pd

from . import config, features, train


def _market_neutral_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """市场中性标签：减去同一时间截面的均值，变成「相对全场」的收益（更适合池化/山寨币）。"""
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

    panel = features.build_panel(progress=False)
    panel = _market_neutral_labels(panel)
    feats = train.feature_cols(panel)
    print(f"[sweep] 面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 币 | {len(feats)} 特征")

    # 目标：原始收益 vs 市场中性收益；阈值：下单的截面 z 门槛
    targets = [("原始收益", ""), ("市场中性", "_mn"), ("截面z", "_cs")]
    zs = [0.5, 1.0, 1.5]

    rows = []
    for h in config.HORIZONS:
        for tname, suffix in targets:
            label_col = f"fwd_ret_{h}h{suffix}"
            for z in zs:
                res = train.evaluate(panel, feats, h, label_col, z, ret_col=f"fwd_ret_{h}h")
                if res is None:
                    continue
                m = res["metrics"]
                mg = res["metrics_gross"]
                rows.append({
                    "horizon_h": h, "target": tname, "z": z,
                    "ic": res["ic"],
                    "sharpe": m["sharpe"], "sharpe_gross": mg["sharpe"],
                    "total_ret": m["total_ret"],
                    "max_dd": m["max_dd"], "win_rate": m["win_rate"],
                    "pf": m["pf"], "n_steps": m["n"],
                    "long_n": res["long_n"], "short_n": res["short_n"],
                    "n_folds": res["n_folds"],
                })
                print(f"[sweep] {h:>2}h {tname:<4} z={z:<3} -> IC={res['ic']:+.4f} "
                      f"毛Sharpe={mg['sharpe']:+.2f} 净Sharpe={m['sharpe']:+.2f} 收益={m['total_ret']:+.1%} 回撤={m['max_dd']:.1%}")

    results = pd.DataFrame(rows)
    if results.empty:
        print("[sweep] 无有效结果")
        return

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.OUTPUT_DIR / "sweep_results.csv", index=False)

    # 排名榜：IC 是核心（不依赖阈值调参的"有没有预测力"），Sharpe 次要
    ranked = results.sort_values("ic", ascending=False)
    print("\n=== 排名榜（按池化 IC 降序，IC 是不受 z 调参影响的预测力指标） ===")
    cols = ["horizon_h", "target", "z", "ic", "sharpe", "total_ret", "max_dd", "pf", "n_steps"]
    print(ranked[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\n[sweep] 共 {len(results)} 个配置，结果已存 {config.OUTPUT_DIR / 'sweep_results.csv'}")


if __name__ == "__main__":
    main()