# -*- coding: utf-8 -*-
"""组合构建器：把「预测」翻译成「下注」。决策层核心。

pred_z → 仓位，含：
- 信号门槛（|z| ≥ threshold 才持有）
- 权重映射（w ∝ |z|^z_power，1=线性、2=凸）
- 波动率目标（vol targeting，w ∝ 1/vol，可选）
- 敞口上限（单边 gross 上限 + 单币上限）
- 状态门（截面分化度不足 → 空仓，即「非友好状态」的一种）

纯函数、无状态，参数集中在 PORTFOLIO_PARAMS，可批量扫（横向吞吐）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PORTFOLIO_PARAMS = {
    "z_threshold": 1.0,      # 下单门槛：|pred_z| ≥ 此值才持有
    "z_power": 1.0,          # 权重映射幂次：w ∝ |z|^z_power
    "risk_budget": 0.02,     # 单笔最大亏损 = 总资金 2%（>0 时启用风险预算反推仓位，替代等权）
    "atr_mult": 2.0,         # 止损距离 = atr_mult × ATR
    "atr_col": "atr14_norm", # 归一化 ATR 列名（features 已算）
    "vol_target": 0.0,       # 波动率目标（0=关闭；>0 时按 1/vol 缩放）
    "vol_col": "vol_24h",    # 波动率列名（paper 面板特征里自带）
    "max_gross": 2.0,        # 总敞口上限（多头合计 + 空头合计）
    "max_single": 0.25,      # 单币权重上限
    "min_signals": 6,        # 截面至少 N 个币有信号(|z|≥1)才下注，否则截面分化不足 → 空仓
    "max_vol_regime": 0.03,  # 高波动 regime：vol_24h 中位数 > 3% → 高成本噪音 → 空仓
    "min_trend_eff": 0.10,   # 震荡 regime：trend_eff_48 中位数 < 0.10 → 剧烈震荡 → 空仓
    "short_max_ret24": 0.20,  # 做空币 24h涨跌幅上限：|24h涨跌|≥20% 的单边暴涨币禁止做空（插针/资金费/下架）
    "regime_action": "flat", # 非友好状态动作：flat=空仓 / half=减半
}


def build_weights(cur: pd.DataFrame, params: dict | None = None) -> tuple[dict, pd.DataFrame, dict]:
    """pred_z → 仓位。返回 (weights, X, meta)。

    weights = {symbol: weight}（美元中性，多头合计 +1、空头合计 -1 再受敞口约束）
    X = cur 转成 symbol 索引、带 price/rank（兼容 paper 的 live IC 结算）
    meta = 状态门信息（n_sig / state / gross），供审计与批量扫
    """
    p = {**PORTFOLIO_PARAMS, **(params or {})}
    X = cur.set_index("symbol").copy()
    if "price" not in X.columns:
        X["price"] = X["close"]
    X["rank"] = X["pred_z"]

    # —— 状态门：三种非友好状态（截面分化不足 / 高波动 / 剧烈震荡）——
    # 注：原实现用 std(pred_z) 测「截面分化度」是坏的——pred_z 已截面标准化，std 恒=1，
    #     测不出分化。改用「有信号的币数」：|z|≥1 的币太少 = 截面没分化，做多空对冲无意义。
    n_sig = int((X["pred_z"].abs() >= p["z_threshold"]).sum()) if len(X) > 1 else 0
    vol_median = float(X[p["vol_col"]].median()) if p["vol_col"] in X.columns else 0.0
    trend_median = float(X["trend_eff_48"].median()) if "trend_eff_48" in X.columns else 1.0
    regime = "normal"
    if n_sig < p["min_signals"]:
        regime = "low_dispersion"
    elif p["max_vol_regime"] > 0 and vol_median > p["max_vol_regime"]:
        regime = "high_vol"
    elif p["min_trend_eff"] > 0 and trend_median < p["min_trend_eff"]:
        regime = "choppy"
    if regime != "normal" and p["regime_action"] == "flat":
        return {}, X, {"n_sig": n_sig, "vol_median": vol_median,
                       "trend_median": trend_median, "state": regime}

    # —— 信号过滤：|z| ≥ 门槛，多空两侧都要有 ——
    sig = X[X["pred_z"].abs() >= p["z_threshold"]]
    longs = sig[sig["pred_z"] > 0]
    shorts = sig[sig["pred_z"] < 0]
    # 动作 A：单边暴涨币禁止做空（|24h涨跌|≥阈值：插针/资金费/下架）。做多不限（赔率有利，赢大输小）。
    if p["short_max_ret24"] > 0 and "ret_24h" in X.columns:
        short_ret = np.expm1(X.loc[shorts.index, "ret_24h"]).abs()
        shorts = shorts[short_ret < p["short_max_ret24"]]
    if longs.empty or shorts.empty:
        return {}, X, {"n_sig": n_sig, "state": "flat"}

    # —— 仓位：风险预算反推（仓位 = 风险预算 ÷ 止损距离），或等权 |z|^power ——
    weights: dict[str, float] = {}
    if p["risk_budget"] > 0 and p["atr_col"] in X.columns:
        # 波动率自适应：每币仓位 = 风险预算 ÷ (k×ATR)，高波动小仓、低波动大仓，
        # 保证「每笔最大亏损 = 仓位 × 止损距离 = 风险预算」恒定。
        atr = X[p["atr_col"]].clip(lower=0.005)
        lpos = (p["risk_budget"] / (p["atr_mult"] * atr[longs.index])).clip(upper=p["max_single"])
        spos = (p["risk_budget"] / (p["atr_mult"] * atr[shorts.index])).clip(upper=p["max_single"])
        lsum, ssum = float(lpos.sum()), float(spos.sum())
        if lsum > 0 and ssum > 0:
            neutral = min(lsum, ssum)   # 美元中性：两侧敞口相等，取较小侧
            lpos = lpos * neutral / lsum
            spos = spos * neutral / ssum
            weights.update((s, float(w)) for s, w in lpos.items())
            weights.update((s, float(-w)) for s, w in spos.items())
    else:
        # 等权 |z|^power（risk_budget=0 时的旧口径）
        lw = longs["pred_z"].abs() ** p["z_power"]
        sw = shorts["pred_z"].abs() ** p["z_power"]
        if lw.sum() > 0:
            weights.update((s, float(w)) for s, w in (lw / lw.sum()).items())
        if sw.sum() > 0:
            weights.update((s, float(-w)) for s, w in (sw / sw.sum()).items())

    # —— 波动率目标：w ∝ 1/vol，再整体缩放到目标总敞口 ——
    if p["vol_target"] > 0 and p["vol_col"] in X.columns:
        vol = X[p["vol_col"]].clip(lower=1e-6)
        w = pd.Series(weights)
        raw = w / vol[w.index]
        denom = raw.abs().sum()
        if denom > 0:
            raw = raw * p["vol_target"] / denom
            weights = raw.to_dict()

    # —— 敞口上限 + 单币上限 ——
    gross = sum(abs(w) for w in weights.values())
    if gross > p["max_gross"]:
        scale = p["max_gross"] / gross
        weights = {s: w * scale for s, w in weights.items()}
    weights = {s: max(-p["max_single"], min(p["max_single"], w)) for s, w in weights.items()}

    # —— 状态门 half：非友好状态减半（flat 已在前面的空仓返回）——
    if regime != "normal" and p["regime_action"] == "half":
        weights = {s: w * 0.5 for s, w in weights.items()}

    meta = {"n_sig": n_sig, "vol_median": vol_median,
            "trend_median": trend_median, "state": regime,
            "gross": sum(abs(w) for w in weights.values())}
    return weights, X, meta
