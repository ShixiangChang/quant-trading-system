# -*- coding: utf-8 -*-
"""时间效应诊断：删时间特征，看 IC 掉不掉。

hour_sin/hour_cos/dow_sin/dow_cos 是时间哑变量，之前占重要性前列（危险信号）。
诊断：同一特征集、同一目标，跑「含时间特征」vs「删时间特征」两版 1-seed 回测，对比 IC/净 Sharpe。
- 删了 IC 明显掉 → 时间在代理真东西（要找它代理的变量显式建模）
- 删了 IC 不变 / 更好 → 时间特征纯过拟合，该删

用法: python -m model.time_diag
"""
from __future__ import annotations

import sys

from . import config, features, train

TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
HORIZON = 96


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print(f"[time_diag] pool {len(pool)} 币，构建面板（use_micro=False 干净对比）…")
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    feats_all = train.feature_cols(panel)
    time_in = [c for c in TIME_COLS if c in feats_all]
    feats_no_time = [c for c in feats_all if c not in TIME_COLS]
    print(f"[time_diag] 特征 {len(feats_all)} 个，其中时间特征 {len(time_in)} 个：{time_in}")

    # 1-seed 快扫（吞吐量优先，诊断只要方向）
    old_seeds = config.ENSEMBLE_SEEDS
    config.ENSEMBLE_SEEDS = 1
    try:
        label = f"fwd_ret_{HORIZON}h_cs"
        r_with = train.evaluate(panel, feats_all, HORIZON, label, 1.0, ret_col=f"fwd_ret_{HORIZON}h")
        r_without = train.evaluate(panel, feats_no_time, HORIZON, label, 1.0, ret_col=f"fwd_ret_{HORIZON}h")
    finally:
        config.ENSEMBLE_SEEDS = old_seeds

    def brief(name, r):
        if r is None:
            print(f"  {name}: 无有效折")
            return None
        m = r["metrics"]
        print(f"  {name}: IC={r['ic']:+.4f} | 净Sharpe={m['sharpe']:+.2f} | "
              f"收益={m['total_ret']:+.1%} | 回撤={m['max_dd']:.1%}")
        return r["ic"]

    print("\n=== 时间效应诊断（1-seed 快扫，96h 截面z） ===")
    ic_with = brief("含时间特征", r_with)
    ic_without = brief("删时间特征", r_without)

    if ic_with is not None and ic_without is not None:
        delta = ic_without - ic_with
        if delta < -0.005:
            verdict = "IC 明显下降 → 时间在代理真东西，别删，要找它代理的变量显式建模"
        elif abs(delta) <= 0.005:
            verdict = "IC 基本不变 → 时间特征贡献≈0，可删（省 4 个噪声特征）"
        else:
            verdict = "IC 反而上升 → 时间特征是纯过拟合，立刻删"
        print(f"\n[time_diag] ΔIC={delta:+.4f} → {verdict}")


if __name__ == "__main__":
    main()
