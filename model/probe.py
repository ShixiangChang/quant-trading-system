# -*- coding: utf-8 -*-
"""扩池探针：用自定义池（probe_pool.txt）跑 IC，验证「覆盖长尾暴涨币」后预测力几何。

关键问题一句话：把截面选币从 18 个蓝筹扩到含长尾暴涨标的的池，截面 z 信号
还有没有 edge（IC 是否维持正、净 Sharpe 是否仍覆盖成本）。有 edge → 铺全市场几百币。

用法（在项目根目录下）:
    python -m model.probe
"""
from __future__ import annotations

import sys
import time

from . import config, features, train


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pool_file = sys.argv[1] if len(sys.argv) > 1 else "probe_pool.txt"
    pool = (config.OUTPUT_DIR / pool_file).read_text(encoding="utf-8").split()
    print(f"[probe] 池 {len(pool)} 币，构建面板（历史 K 线 + 费率）…")
    panel = features.build_panel(progress=True, symbols=pool)
    feats = train.feature_cols(panel)
    print(f"[probe] 面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 币 | {len(feats)} 特征")

    # 关键配置：截面 z 标签（与 decision 一致），记账用原始收益。z=1.0
    for h in (12, 24, 48):
        label = f"fwd_ret_{h}h_cs"
        ret = f"fwd_ret_{h}h"
        t0 = time.time()
        res = train.evaluate(panel, feats, h, label, 1.0, ret_col=ret)
        if res is None:
            print(f"[probe] {h:>2}h 无有效折，跳过")
            continue
        m, mg = res["metrics"], res["metrics_gross"]
        print(f"[probe] {h:>2}h 截面z z=1.0: IC={res['ic']:+.4f} "
              f"净Sharpe={m['sharpe']:+.2f} 毛Sharpe={mg['sharpe']:+.2f} "
              f"收益={m['total_ret']:+.1%} 回撤={m['max_dd']:.1%} "
              f"折数={res['n_folds']} 多{res['long_n']} 空{res['short_n']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()