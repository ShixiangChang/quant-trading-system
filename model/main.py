# -*- coding: utf-8 -*-
"""入口。

用法（在项目根目录下）:
    python -m model.main
"""
from __future__ import annotations

import sys

from . import config, features, train


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("建模：历史数据 → 特征 → LightGBM walk-forward 回测")
    print(f"[main] 宇宙 {len(config.UNIVERSE)} 币种 | 周期 {config.INTERVAL} | 历史 {config.DAYS} 天")

    panel = features.build_panel(progress=True)
    print(f"[main] 面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 币种 | "
          f"{len(train.feature_cols(panel))} 特征 | 起止 "
          f"{panel['open_time'].min():%Y-%m-%d} ~ {panel['open_time'].max():%Y-%m-%d}")

    train.walk_forward(panel)


if __name__ == "__main__":
    main()