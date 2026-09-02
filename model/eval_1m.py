# -*- coding: utf-8 -*-
"""1h 因子 vs 1h+1m 因子 walk-forward 对比。

回答一个问题：1m 精细因子（尤其 rv_24h）喂给 LightGBM 预测未来 96h 截面收益，
是否比「只用 1h 因子」有增量。用 18 个 1m 完整币做公平对比池。

结论口径：核心看 IC（预测值 vs 标签的截面 rank 相关），不受截面大小 / z 阈值干扰。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config, features, features_1m, train

SYMBOLS = features_1m.FULL_1M_SYMBOLS
LABEL = "fwd_ret_96h_cs"     # 截面 z 化标签（模型学截面相对强弱，非原始收益）
RET = "fwd_ret_96h"          # 记账用真实收益


def _report(tag: str, r: dict) -> dict:
    m = r["metrics"]
    ic = r["ic"]
    ic_fold = float(np.mean(r["ic_per_fold"])) if r["ic_per_fold"] else 0.0
    daily = r["daily_ic"]
    daily_mean = float(daily.mean()) if len(daily) else 0.0
    line = (f"[{tag}] IC(pool)={ic:+.4f} | IC(折均)={ic_fold:+.4f} | "
            f"IC(日均)={daily_mean:+.4f} | Sharpe={m['sharpe']:+.2f} | "
            f"总收益={m['total_ret']:+.1%} | 回撤={m['max_dd']:+.1%} | "
            f"胜率={m['win_rate']:.1%} | 步数={m['n']}")
    print(line)
    return {"ic_pool": ic, "ic_fold_mean": ic_fold, "ic_daily_mean": daily_mean,
            "sharpe": m["sharpe"], "total_ret": m["total_ret"], "max_dd": m["max_dd"],
            "win_rate": m["win_rate"], "n": m["n"]}


def main():
    # 本轮聚焦纯价量 + 1m 增量，临时禁用 funding/micro（两组一致，公平）
    config.USE_FUNDING = False

    print("=== 建 1h 面板（18 币，纯价量） ===")
    panel = features.build_panel(progress=True, symbols=SYMBOLS, use_micro=False)
    feats_1h = train.feature_cols(panel)
    print(f"1h panel: {len(panel):,} 行 × {panel['symbol'].nunique()} 币 | 1h 特征 {len(feats_1h)} 个")

    print("\n=== 建 1m 因子并 merge ===")
    f1m = features_1m.build_1m_factors(SYMBOLS)
    panel_full = features_1m.merge_1m(panel, f1m)
    feats_full = train.feature_cols(panel_full)
    added = [c for c in feats_full if c not in feats_1h]
    print(f"新增 1m 因子 {len(added)} 个: {added}")
    print(f"1m 因子覆盖: {panel_full['rv_24h_1m'].notna().mean():.1%} 行")

    # sanity check：rv_24h_1m 对未来 96h 收益的逐截面 rank IC，应 ≈ -0.09（对齐正确性验证）
    sc = panel_full.dropna(subset=["rv_24h_1m", "fwd_ret_96h"])
    ics = []
    for _, g in sc.groupby("open_time"):
        if len(g) >= 8:
            ics.append(g["rv_24h_1m"].corr(g["fwd_ret_96h"], method="spearman"))
    print(f"[sanity] rv_24h_1m → 未来96h 逐截面 rank IC = {np.mean(ics):+.4f}（应≈-0.09）")

    print(f"\n=== walk-forward：label={LABEL}, horizon=96, z=±1.0 ===")
    base = train.evaluate(panel, feats_1h, horizon=96, label_col=LABEL, z=1.0, ret_col=RET)
    enh = train.evaluate(panel_full, feats_full, horizon=96, label_col=LABEL, z=1.0, ret_col=RET)

    r_base = _report("baseline(1h)", base)
    r_enh = _report("enhanced(1h+1m)", enh)

    # 特征重要性（全数据训一棵，看 1m 因子排位）
    print("\n=== 特征重要性 Top 15（gain，含 1m 因子） ===")
    d = lgb.Dataset(panel_full[feats_full].to_numpy(dtype=float),
                    label=panel_full[LABEL].to_numpy(dtype=float))
    b = lgb.train(dict(config.LGBM_PARAMS), d, num_boost_round=config.NUM_BOOST_ROUND)
    imp = pd.Series(b.feature_importance("gain"), index=feats_full).sort_values(ascending=False)
    top = {}
    for feat, val in imp.head(15).items():
        flag = "  <<1m" if feat.endswith("_1m") else ""
        print(f"  {feat:<22s} {val:,.0f}{flag}")
        top[feat] = float(val)

    payload = {
        "symbols": SYMBOLS, "label": LABEL, "ret": RET, "horizon": 96,
        "n_1h_feats": len(feats_1h), "n_full_feats": len(feats_full),
        "added_1m_feats": added,
        "baseline": r_base, "enhanced": r_enh,
        "top_importance": top,
        "rv24_ic_sanity": float(np.mean(ics)),
    }
    out = config.OUTPUT_DIR / "eval_1m.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已落盘 {out}")


if __name__ == "__main__":
    main()
