# -*- coding: utf-8 -*-
"""多口径扫描报告：批量保留 + 多口径排序。

读 sweep 结果 CSV（默认 focused_sweep.csv，去泄漏 12 配置），输出多个排序视角——
全局榜、预测力榜、每周期最优、每目标最优、山脊检测、稳健性。让选择变成
「从多个视角看同一批结果」，识别山脊 vs 孤峰，不预设唯一。

用法: python -m model.report [csv_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import config

DEFAULT_CSV = config.OUTPUT_DIR / "focused_sweep.csv"
_COLS = ["horizon_h", "target", "ic", "sharpe", "total_ret", "max_dd", "n_steps"]


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:+.3f}")


def multi_view(df: pd.DataFrame) -> None:
    # ① 全局榜：净 Sharpe（单一指标，仅作视角之一）
    print("\n=== ① 全局榜 · 按净 Sharpe 降序 ===")
    print(_fmt(df.sort_values("sharpe", ascending=False)[_COLS]))

    # ② 预测力榜：IC（不受 z 阈值调参影响的「有没有预测力」）
    print("\n=== ② 预测力榜 · 按 IC 降序 ===")
    print(_fmt(df.sort_values("ic", ascending=False)[_COLS]))

    # ③ 每周期最优
    print("\n=== ③ 每周期最优（该 horizon 下净 Sharpe 最高的目标） ===")
    best_h = df.loc[df.groupby("horizon_h")["sharpe"].idxmax(), _COLS]
    print(_fmt(best_h.sort_values("horizon_h")))

    # ④ 每目标最优
    print("\n=== ④ 每目标最优（该目标下净 Sharpe 最高的周期） ===")
    best_t = df.loc[df.groupby("target")["sharpe"].idxmax(), _COLS]
    print(_fmt(best_t))

    # ⑤ 山脊检测：每个目标在 horizon 序列上的 Sharpe 轨迹（识别孤峰 vs 山脊）
    print("\n=== ⑤ 山脊检测（每个目标 × horizon 的 Sharpe 轨迹） ===")
    for tgt, g in df.groupby("target"):
        g = g.sort_values("horizon_h")
        seq = "  ".join(f"{int(r.horizon_h)}h:{r.sharpe:+.2f}" for r in g.itertuples())
        print(f"  {tgt:<6} {seq}")

    # ⑥ 稳健性：正 Sharpe 的周期占比（越高越不依赖单一周期）
    print("\n=== ⑥ 稳健性（正 Sharpe 的周期数 / 总周期数） ===")
    for tgt, g in df.groupby("target"):
        pos = int((g["sharpe"] > 0).sum())
        print(f"  {tgt:<6} {pos}/{len(g)} 个周期为正")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not path.exists():
        print(f"[report] 找不到 {path}，先跑 python -m model.sweep 生成结果")
        return
    df = pd.read_csv(path)
    print(f"[report] 读 {path}：{len(df)} 个配置 | {df['target'].nunique()} 目标 | "
          f"horizon={sorted(df['horizon_h'].unique())}")
    multi_view(df)


if __name__ == "__main__":
    main()
