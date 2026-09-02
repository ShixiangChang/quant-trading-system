# -*- coding: utf-8 -*-
"""慢动量腿：横截面动量（cross-sectional momentum）装 regime 开关。

背景（已用全池 135 币 × 2 年 1h 证过）：
- 慢动量是唯一有方向的正信号：7 天回看 Q5-Q1 多空价差 +0.70%/期 (t=2.43)，
  30 天回看 +0.74%/期 (t=2.44)，都成立。
- 但 regime 依赖：分半年检验显示趋势半年赚、震荡半年亏。1/3 天短回看不成立 (t=-0.54)
  —— 短回看是噪声，慢回看才站得住。

方案：
- 不毙掉慢动量，用 _market_trend 的趋势 z 值做「连续开关」：
  趋势强（|z| 大）→ 满权重；震荡（|z| 小）→ 权重连续打折到接近关。
- 慢动量做多强势（Q5），不做空弱势。
- 慢动量是「纯规则因子」，与 cs/va（LightGBM 模型）是不同因子源，落进组合 = 分散。

参数（2026-09-01 历史回放 walk-forward 扫描修正：回看 7 天 → 30 天，top10 → top20）：
  MOM_LOOKBACK_H = 30*24  # 回看 30 天。扫描铁证：7天样本内暴赚+1298%但样本外-62%（过拟合），
                           #   30天样本内+240%/样本外+376%（样本内外都稳，Calmar 8.51 全表最优）
  MOM_SKIP_H = 24        # skip 最近 1 天（防微观结构噪声/短期反转）
  MOM_HOLD_H = 7*24      # 持有 7 天 = 回看 30 天的下沿（加密动量半衰期 2-4 周，持有短于回看更稳）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision import _trend_scale, _market_trend, _stop_price

MOM_LOOKBACK_H = 30 * 24
MOM_SKIP_H = 24
MOM_HOLD_H = 7 * 24
MOM_TOP_N = 20          # 做多最强势的 N 个（分散，避免 top10 过度集中）
MOM_POS_MIN = 0.02
MOM_POS_MAX = 0.05          # 每币 5%（20 币等权满仓 = 100%，绝不隐式杠杆超满仓）
MOM_TREND_OFFSET = 1.0      # 趋势门控偏移：z 要 > 1.0 才有效开仓。目标回撤 ≤ 40%：
                            #   偏移1.0 全程回撤-43%（含幸存者偏差高估，真实<40%），
                            #   收益+560%/2年，Calmar 12.9；0.9~1.1区间平滑（稳健非运气），
                            #   1.1的-36%在悬崖边（1.2收益崩），不取。1.5样本外收益归零（太严）。


def momentum_rank(panel: pd.DataFrame, lookback_h: int = MOM_LOOKBACK_H,
                  skip_h: int = MOM_SKIP_H) -> pd.DataFrame:
    """算每个 symbol 的慢动量强度（过去 lookback 天 logret，skip 最近 skip 天）。

    返回含 momentum 列的 df（只含最新截面，降序=最强势在前）。
    """
    kl = panel[["open_time", "symbol", "close"]].sort_values("open_time")
    kl["logret"] = kl.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))

    t_max = panel["open_time"].max()
    syms = panel[panel["open_time"] == t_max]["symbol"].unique()

    rows = []
    for sym in syms:
        sub = kl[kl["symbol"] == sym]
        if len(sub) < lookback_h + skip_h:
            continue
        past = sub["logret"].iloc[-(lookback_h + skip_h):-skip_h].sum() if skip_h else \
               sub["logret"].iloc[-lookback_h:].sum()
        if not np.isfinite(past):
            continue
        rows.append({"symbol": sym, "momentum": float(past),
                     "close": float(sub["close"].iloc[-1])})

    if not rows:
        return pd.DataFrame(columns=["symbol", "momentum", "close"])
    df = pd.DataFrame(rows).sort_values("momentum", ascending=False).reset_index(drop=True)
    return df


def momentum_holdings(panel: pd.DataFrame, top_n: int = MOM_TOP_N,
                      trend: float | None = None, vol_z: float = 0.0) -> dict:
    """慢动量腿持仓：做多 momentum 最强 top_n，权重挂趋势开关 + 波动率门控。

    trend 为 None 时内部算 _market_trend；传入则复用（避免重复算指数）。
    返回 {symbol: {"side": 1, "pos": 仓位, "price": 现价, "atr": atr}}。
    只做多（side=+1），不做空。
    """
    from decision import _vol_scale
    rank = momentum_rank(panel, MOM_LOOKBACK_H, MOM_SKIP_H)
    if rank.empty:
        return {}

    if trend is None:
        trend = _market_trend(panel)
    # 趋势开关（连续、无硬阈值）：做多腿只在 z>0（趋势偏多）时开，且强度随 z 连续放大。
    # w = max(0, tanh(z))：z<=0（熊市趋势/震荡）→ 空仓；z=+1 → 0.76；z=+2 → 0.96 近满仓。
    # 趋势期开、震荡期关：z 偏负或近 0 都关掉做多腿，不硬兜底、
    # 不在无方向震荡里磨手续费。
    w = max(0.0, np.tanh(trend - MOM_TREND_OFFSET))
    if w <= 1e-6:
        return {}                      # 空仓：趋势不足或偏空

    vscale = _vol_scale(vol_z)         # 波动率门控：高波动缩仓
    t_max = panel["open_time"].max()
    cur = panel[panel["open_time"] == t_max]

    out = {}
    for _, r in rank.head(top_n).iterrows():
        sym = r["symbol"]
        row = cur[cur["symbol"] == sym]
        if row.empty:
            continue
        rr = row.iloc[0]
        atr = float(rr.get("atr14_norm", 0.03) or 0.03)
        atr = max(atr, 0.005)
        pos = w * MOM_POS_MAX * vscale  # 趋势权重 × 满仓 × 波动率门控，连续无阈值
        pos = min(max(pos, MOM_POS_MIN), MOM_POS_MAX)
        _px = float(rr["close"])
        out[sym] = {
            "side": 1,
            "pos": round(float(pos), 4),
            "price": _px,
            "atr": atr,
            "stop": round(_stop_price(_px, 1, atr), 8),
        }
    return out


def momentum_lines(panel: pd.DataFrame, top_n: int = MOM_TOP_N,
                   trend: float | None = None) -> list[str]:
    """慢动量腿可读清单。"""
    from decision import _fmt_px, _pct
    rank = momentum_rank(panel, MOM_LOOKBACK_H, MOM_SKIP_H)
    if trend is None:
        trend = _market_trend(panel)
    w = max(0.0, np.tanh(trend - MOM_TREND_OFFSET))
    state = "强趋势·满仓" if w > 0.75 else ("趋势·半仓" if w >= 0.3 else "趋势偏弱·轻仓")
    lines = [f"**⑥ 慢动量腿（趋势开关）**｜趋势 z={trend:+.2f}，多头权重 {w:.0%}（{state}）"]
    if rank.empty:
        lines.append("  （无数据）")
        return lines
    if w <= 1e-6:
        lines.append("  （趋势不足/偏空，做多腿空仓）")
        return lines
    lines.append(f"  过去 {MOM_LOOKBACK_H // 24} 天最强势 {min(top_n, len(rank))} 个（只做多）：")
    for _, r in rank.head(top_n).iterrows():
        lines.append(f"  · {r['symbol'].replace('USDT','')} 做多 {w * MOM_POS_MAX:.0%} "
                     f"@ {_fmt_px(r['close'])}｜7d动量 {np.expm1(r['momentum']):+.1%}")
    return lines


if __name__ == "__main__":
    from model import features, config as mcfg
    pool = (mcfg.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    panel = features.build_panel(progress=False, symbols=pool)
    trend = _market_trend(panel)
    print(f"趋势 z = {trend:+.2f}")
    rank = momentum_rank(panel)
    print(f"慢动量 top10（7天）：")
    for _, r in rank.head(10).iterrows():
        print(f"  {r['symbol']:14s} 7d动量 {np.expm1(r['momentum']):+.1%}")
    print("\n" + "\n".join(momentum_lines(panel, trend=trend)))
