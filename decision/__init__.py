# -*- coding: utf-8 -*-
"""决策机器人：训练 ML「截面 z 24h」信号 → 生成「今日操作清单」→ 推钉钉。

系统输出 = 今日哪 N 个币值得操作：方向 + 开仓/止盈/止损价 + 梯度规则 + 理由。
每笔预测落库，24h 后自动对账结算命中率，形成监督学习闭环。

用法（在项目根目录下）:
    python -m decision             # 拉最新数据 → 训练 → 预测 → 出清单 → 推钉钉
    python -m decision --dry       # 用现有缓存，只打印清单，不推送
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from model import config as mcfg, features, train, data as mdata, market_data as md
from monitor import config as ncfg
from monitor.db import MonitorDB

HORIZON = 96
Z = 1.0                  # 截面 z 阈值（训练过滤用）
TOP_N = 10               # 每日交付的操作币数量
POOL_FILE = "probe_pool.txt"   # 操作池（57 币：Top50 流动性 ∪ 暴涨暴跌 mover）
ATR_MULT = 3.0           # 止损距离 = 3 × ATR(14)（给足空间，不被噪音扫掉）
LIMIT_ATR_MULT = 1.0      # 限价入场距离 = 1 × ATR：做多等回调 1 ATR 再买，做空等反弹 1 ATR 再空
                          # 方向对但位置差时，用限价单等价格回到更优位置才成交；不成交则不做
RISK_PER_TRADE = 0.02    # 单笔最大亏损 = 总资金 2%（风险预算反推仓位用）
CORE_VOL_FLOOR = 100_000_000   # 已废弃（2026-09-01）：旧「成交额≥1亿切大币」二元标签，被 model/market_data.py 取代
                              # （市值分位数定规模 + 流动性门槛定可交易 + 换手率连续特征）
CORE_SAT_RATIO = 0.7           # 已废弃（组合腿 2026-09-01 砍掉，卫星腿并入动量腿）
SHORT_MAX_RET24 = 0.20        # 动作 A：|24h涨跌幅| ≥ 20% 的币禁止做空（单边暴涨币，插针/资金费/下架）
STRATEGY_STATE_FILE = "strategy_state.json"
RUN_INTERVAL_H = 1        # 常驻检查间隔（小时）：每小时重算信号，但只在「信号有变化」时才推送
RUN_HOUR_UTC = 0           # 每日决策运行时刻（UTC 00:30 = 北京 08:30）——已由 RUN_INTERVAL_H 取代，保留兼容
RUN_MINUTE_UTC = 30

# —— 多目标 × 多 horizon 并行 ——
# 注：mn（市场中性）目标已废弃——该标签=收益减截面均值，在 crypto 里方差极小接近白噪声，
# LightGBM 学不动（66% 预测值近零），方向退化为常做空，是纸面仓失血的主因之一。已删除。
TARGETS = [
    {"key": "cs", "name": "截面z",   "suffix": "_cs"},   # 截面标准化收益（相对全场）
    {"key": "va", "name": "波动率调整", "suffix": "_va"}, # 超额收益÷币自身波动率（GRJMOM，消除波动率偏差）
]
HORIZONS = [24, 48, 96]    # crypto 自然节奏：24h情绪日 / 48h动量 / 96h趋势（多窗口融合，不押单一窗口）

# —— 市场状态（动作 E：Trend Following 趋势过滤，Faber 2007 / CTA 行业标准）——
# 全池等权中位数指数 vs 慢均线的「标准化偏离 z」（偏离率 ÷ 其历史标准差）：连续量、无离散阈值、参数自标定。
# 净敞口偏置 w = 0.5 ± 0.5·tanh(z)：行情越强偏置越极端（极端行情 ≠ 普通行情），但方向腿永不关死——
# 截面信号「多强空弱」与牛熊无关，牛市也有该做空的弱势币、熊市也有该做多的强势币。
TREND_MA_HOURS = 720       # 慢均线：30 日（1h 数据 = 30×24 根）


def _load_strategy() -> dict:
    p = mcfg.OUTPUT_DIR / STRATEGY_STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"top_n": TOP_N, "adjustments": []}


def _save_strategy(st: dict) -> None:
    mcfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (mcfg.OUTPUT_DIR / STRATEGY_STATE_FILE).write_text(
        json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def _self_heal(st: dict, rep: dict) -> tuple[int, str]:
    """前向自愈：只用真实结算的期望值调 TOP_N（输了→减仓降风险，稳赚→恢复），全程可审计。

    规则（保守，只按前向真实结算，绝不按回测调）：
    - 结算 ≥ 20 笔且期望值 < 0 → 减 3 个仓（下限 3）。
    - 结算 ≥ 40 笔且期望值 > +0.5%/笔 → 恢复 +3（上限 15）。
    - 其余不动。
    """
    n, exp = rep["n"], rep["expectancy"]
    top = int(st.get("top_n", TOP_N))
    new_top, reason = top, ""
    if n >= 20 and exp < 0 and top > 3:
        new_top = max(3, top - 3)
        reason = f"结算 {n} 笔期望 {exp:+.2%}/笔<0 → 减仓 {top}→{new_top}"
    elif n >= 40 and exp > 0.005 and top < 15:
        new_top = min(15, top + 3)
        reason = f"结算 {n} 笔期望 {exp:+.2%}/笔>0.5% → 加仓 {top}→{new_top}"
    if new_top != top:
        st["adjustments"].append({"ts": int(time.time()), "from": top, "to": new_top, "reason": reason})
    st["top_n"] = new_top
    return new_top, reason


def _fit_predict(tr, te, feats, label_col):
    """5-seed 集成预测取平均（压掉 big-model 单棵树方差）。"""
    dtr = lgb.Dataset(tr[feats].to_numpy(dtype=float), label=tr[label_col].to_numpy(dtype=float))
    Xte = te[feats].to_numpy(dtype=float)
    preds = []
    for k in range(mcfg.ENSEMBLE_SEEDS):
        params = dict(mcfg.LGBM_PARAMS)
        params["seed"] = mcfg.LGBM_PARAMS["seed"] + k
        b = lgb.train(params, dtr, num_boost_round=mcfg.NUM_BOOST_ROUND)
        preds.append(b.predict(Xte))
    return np.mean(preds, axis=0)


def decide(panel, feats):
    """训练最近 90 天（purge 24h）→ 预测当前最新截面 → 生成信号 + 止损距离。"""
    label_col = f"fwd_ret_{HORIZON}h_cs"
    t_max = panel["open_time"].max()
    purge = pd.Timedelta(hours=HORIZON)
    tr = panel[(panel["open_time"] >= t_max - pd.Timedelta(days=mcfg.TRAIN_DAYS))
               & (panel["open_time"] < t_max - purge)].copy()
    cur = panel[panel["open_time"] == t_max].copy()
    if len(tr) < mcfg.MIN_TRAIN_ROWS or len(cur) < 4:
        return None

    cur["pred"] = _fit_predict(tr, cur, feats, label_col)
    cur["pred_z"] = (cur["pred"] - cur["pred"].mean()) / (cur["pred"].std() + 1e-12)
    cur["sig"] = 0
    cur.loc[cur["pred_z"] >= Z, "sig"] = 1
    cur.loc[cur["pred_z"] <= -Z, "sig"] = -1
    cur["stop_frac"] = (ATR_MULT * cur["atr14_norm"]).clip(lower=0.005)
    return cur, t_max, len(tr)


def _fit_predict_fast(tr, te, feats, label_col):
    """1-seed 快训：用于多目标 × 多周期批量训练；确认阶段使用 5-seed 精算。"""
    dtr = lgb.Dataset(tr[feats].to_numpy(dtype=float), label=tr[label_col].to_numpy(dtype=float))
    Xte = te[feats].to_numpy(dtype=float)
    b = lgb.train(dict(mcfg.LGBM_PARAMS), dtr, num_boost_round=mcfg.NUM_BOOST_ROUND)
    return b.predict(Xte)


def decide_multi(panel, feats):
    """多目标 × 多 horizon 并行预测：每个 (target, horizon) 训练(1-seed) + 预测当前截面。

    返回 {(target_key, horizon): cur_df} 和 t_max。cur_df 含 pred_z + stop_frac。
    """
    t_max = panel["open_time"].max()
    cur_base = panel[panel["open_time"] == t_max].copy()
    results: dict = {}
    for h in HORIZONS:
        purge = pd.Timedelta(hours=h)
        tr = panel[(panel["open_time"] >= t_max - pd.Timedelta(days=mcfg.TRAIN_DAYS))
                   & (panel["open_time"] < t_max - purge)]
        if len(tr) < mcfg.MIN_TRAIN_ROWS or len(cur_base) < 4:
            continue
        for tgt in TARGETS:
            label_col = f"fwd_ret_{h}h{tgt['suffix']}"
            if label_col not in panel.columns:
                continue
            cur = cur_base.copy()
            cur["pred"] = _fit_predict_fast(tr, cur, feats, label_col)
            cur["pred_z"] = (cur["pred"] - cur["pred"].mean()) / (cur["pred"].std() + 1e-12)
            cur["stop_frac"] = (ATR_MULT * cur["atr14_norm"]).clip(lower=0.005)
            results[(tgt["key"], h)] = cur
    return results, t_max


def _index_price(panel: pd.DataFrame) -> pd.Series:
    """全池等权几何指数：每个时间截面所有币 logret 的均值累乘。

    不用 close 中位数（会被币种组成/价格量级污染），用等权收益率的几何累乘（量纲正确）。
    这是「市场 beta」的标准口径，与 beta_backtest.py 一致。
    """
    kl = panel[["open_time", "symbol", "close"]].sort_values("open_time")
    kl["logret"] = kl.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))
    idx_ret = kl.groupby("open_time")["logret"].mean()
    return np.exp(idx_ret.cumsum())


def _market_trend(panel: pd.DataFrame) -> float:
    """市场趋势状态：全池等权几何指数 vs 30 日均线的「标准化偏离 z」（连续量，无阈值）。

    z = (指数/30日均线 - 1) ÷ 该偏离序列的历史标准差。
    z>0 偏多、z<0 偏空，|z| 越大偏离越极端；用自身历史波动自标定「多强算强」，不拍固定阈值。
    """
    idx = _index_price(panel)
    if len(idx) < TREND_MA_HOURS + 1:
        return 0.0
    ma = idx.rolling(TREND_MA_HOURS).mean()
    dev = idx / ma - 1.0          # 全程偏离率序列
    scale = dev.std()             # 历史标准差 → 自标定「多强算强」
    if pd.isna(scale) or scale <= 0:
        return 0.0
    last = float(dev.iloc[-1])
    if pd.isna(last):
        return 0.0
    return last / scale


def _trend_scale(z: float, side: int) -> float:
    """连续净敞口偏置：w = 0.5 ± 0.5·tanh(z)。无离散阈值、无固定数字。

    z = 标准化趋势（指数偏离均线 ÷ 历史std）：
      · z=0（中性）：双边 0.5，净敞口 0（纯市场中性）
      · z=+2（极端牛）：做多 0.98 / 做空 0.02（几乎全多，但不关空腿）
      · z=-2（极端熊）：做多 0.02 / 做空 0.98
      · 程度成比例：|z| 越大偏置越极端，普通行情偏置小（极端 ≠ 普通）
    净敞口 = w_long − w_short = tanh(z)，单一连续量。
    """
    x = np.tanh(z)
    return 0.5 + (0.5 if side > 0 else -0.5) * x


def _market_vol(panel: pd.DataFrame) -> float:
    """全池波动率 z：指数 24h 收益滚动标准差，相对自身历史水平标准化（连续量）。

    高波动（vol_z>0）＝市场剧烈，赔率变差（滑点/插针/止损失效）→ 该降仓离桌；
    低波动（vol_z<0）＝平稳，可正常持仓。用自身历史自标定「多高算高」，不拍固定阈值。
    """
    idx = _index_price(panel)
    if len(idx) < 25:
        return 0.0
    vol = idx.pct_change().rolling(24).std()
    cur = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 0.0
    hist = vol.dropna()
    if len(hist) < 30 or float(hist.std()) <= 0:
        return 0.0
    return (cur - float(hist.mean())) / float(hist.std())


def _vol_scale(vol_z: float) -> float:
    """波动率门控（连续，非二值离桌）：高波动按波动率倒数缩总仓位。

    vol_z≤0（正常/低波动）→ 全仓；vol_z>0 越高缩越多，极端波动（vol_z≥2）缩到 0.5。
    与趋势门控叠加：趋势定「多还是空」，波动定「下多重、还是几乎离桌」。
    """
    z = float(vol_z or 0.0)
    if z <= 0:
        return 1.0
    scale = 1.0 / (1.0 + 0.5 * z)
    return min(max(scale, 0.5), 1.0)


def _beta_position(panel: pd.DataFrame) -> tuple[int, float]:
    """纯 beta 趋势跟随：全池等权几何指数 vs 30 日均线的绝对位置。

    指数 > 均线 → 牛（做多，1）；指数 < 均线 → 熊（空仓，0）。
    不碰截面模型（cs 排序已证伪为负 alpha），只做方向跟随：吃上涨、躲下跌。
    回测实证（过去 1 年熊市）：盲买指数 -48.8%，此策略 30 日线 +10.9% / 180 日线 +27.8%。
    """
    idx = _index_price(panel)
    if len(idx) < TREND_MA_HOURS + 1:
        return 0, 0.0
    ma = idx.rolling(TREND_MA_HOURS).mean()
    last = float(idx.iloc[-1])
    m = ma.iloc[-1]
    if pd.isna(m) or m <= 0:
        return 0, 0.0
    dev = last / m - 1.0
    return (1 if dev > 0 else 0), dev


def beta_holdings(panel: pd.DataFrame, top_n: int = TOP_N) -> dict:
    """beta 腿持仓：牛 → 等权做多全池（按 24h 成交额取 top_n），熊 → 空仓。"""
    direction, _ = _beta_position(panel)
    if direction == 0:
        return {}
    t_max = panel["open_time"].max()
    cur = panel[panel["open_time"] == t_max]
    vol_map = _fetch_quote_vols()
    ranked = sorted(cur["symbol"].unique(), key=lambda s: vol_map.get(s, 0.0), reverse=True)
    out = {}
    pos = 1.0 / max(top_n, 1)
    for s in ranked[:top_n]:
        row = cur[cur["symbol"] == s]
        if row.empty:
            continue
        r = row.iloc[0]
        _atr = float(_pct(r.get("atr14_norm"), 0.03) or 0.03)
        _px = float(r["close"])
        out[s] = {
            "side": 1, "pos": round(pos, 4),
            "price": _px,
            "atr": _atr,
            "stop": round(_stop_price(_px, 1, _atr), 8),
            "z": 0.0,
            "bb_pctb": round(float(_pct(r.get("bb_pctb"), 0.5) or 0.5), 4),
            "bb_width": round(float(_pct(r.get("bb_width"), 0.0) or 0.0), 4),
            "sma20": round(float(_pct(r.get("close_sma20"), 0.0) or 0.0), 4),
            "sma50": round(float(_pct(r.get("close_sma50"), 0.0) or 0.0), 4),
            "d_high48": round(float(_pct(r.get("dist_high48"), 0.0) or 0.0), 4),
            "d_low48": round(float(_pct(r.get("dist_low48"), 0.0) or 0.0), 4),
            "rsi": round(float(_pct(r.get("rsi14"), 50.0) or 50.0), 2),
            "ret24": round(float(np.expm1(_pct(r.get("ret_24h"), 0.0) or 0.0)), 4),
        }
    return out


def _build_report(grid: dict, inserted_n: int, settled_n: int) -> list[str]:
    """吞吐量对账报告：多目标 × 多 horizon 的命中率/期望值矩阵。"""
    lines = [f"**吞吐量对账**：今日落库 {inserted_n} 条 / 结算 {settled_n} 条"]
    lines.append("累计真实结算（|z|≥1 信号仓）：")
    total = 0
    for tgt in TARGETS:
        hs = grid.get(tgt["key"], {})
        for h in sorted(hs):
            c = hs[h]
            total += c["n"]
            lines.append(f"· {tgt['name']}×{h}h：{c['n']} 笔 | 命中 {c['win_rate']:.0%} | 期望 {c['expectancy']:+.2%}/笔")
    if total == 0:
        lines.append("· （暂无结算样本，从今天开始累积）")
    return lines


def _fmt_px(x: float) -> str:
    """价格按量级给小数位 + 千分位：BTC 79290 而非 79,152.1143。"""
    if x >= 1000:
        return f"{x:,.0f}"
    decimals = 2 if x >= 1 else (4 if x >= 0.1 else 5)
    return f"{x:,.{decimals}f}".rstrip("0").rstrip(".")


def _pct(v, default=0.0):
    return float(v) if v is not None and pd.notna(v) else default


def _stop_price(price: float, side: int, atr: float, atr_mult: float = ATR_MULT) -> float:
    """止损价 = 开仓价 × (1 − side × atr_mult × atr)。atr 为归一化比例（如 0.063）。

    做多(side=+1)：stop = price × (1 − 3×atr)，跌破离场；
    做空(side=−1)：stop = price × (1 + 3×atr)，涨破离场。
    """
    atr = max(float(atr or 0.03), 0.005)
    return price * (1.0 - side * atr_mult * atr)


def _pos_weight(side: int, bb_pctb) -> float:
    """入局时机位置过滤（布林带 %b，连续权重，不硬切）。

    均值回归系统的行业标准：做多只在价格位于中轨下方(超跌)时重仓，做空只在中轨上方(超涨)时重仓。
    bb_pctb ∈ [0,1]：0=下轨，0.5=中轨，1=上轨。
      做多 w = clip(1.2 − 1.4×bb_pctb, 0.3, 1.0)  → bb=0 下轨全仓，bb=1 上轨缩到 0.3（不追高）
      做空 w = clip(−0.2 + 1.4×bb_pctb, 0.3, 1.0)  → bb=1 上轨全仓，bb=0 下轨缩到 0.3（不空地板）
    方向对但入场位置差时，位置走过头自动轻仓。
    """
    bb = float(bb_pctb) if bb_pctb is not None else 0.5
    if side > 0:
        w = 1.2 - 1.4 * bb
    else:
        w = -0.2 + 1.4 * bb
    return min(max(w, 0.3), 1.0)


def _reason(r):
    """一句话理由：真实特征快照，措辞跟着「方向 vs 当下涨跌」走，不矛盾。"""
    z = float(r["pred_z"])
    ret24 = float(np.expm1(_pct(r.get("ret_24h"))))
    rsi = float(_pct(r.get("rsi14"), 50.0))
    dhi = float(np.expm1(_pct(r.get("dist_high48"))))
    if z > 0:
        verb = "超卖反弹做多" if ret24 < -0.05 and rsi < 45 else "追强突破"
        return f"截面 z={z:+.1f} 强势第一梯队｜24h {ret24:+.1%}｜距48h高点 {dhi:+.1%}｜RSI {rsi:.0f} → {verb}"
    verb = "暴涨超买见顶做空" if ret24 > 0.05 and rsi > 70 else "弱势延续做空"
    return f"截面 z={z:+.1f} 弱势垫底｜24h {ret24:+.1%}｜RSI {rsi:.0f} → {verb}"


def _funding_cost(symbol: str, start_ts: int, hours: int) -> float:
    """持仓期累计资金费率（符号在调用处按方向处理：做多付、做空收）。无数据按 0 计，不炸对账。"""
    try:
        f = mdata.fetch_funding(symbol)
    except Exception:
        return 0.0
    if f is None or f.empty:
        return 0.0
    start = pd.Timestamp(int(start_ts), unit="s", tz="UTC")
    end = start + pd.Timedelta(hours=int(hours))
    win = f[(f["funding_time"] >= start) & (f["funding_time"] < end)]
    return float(win["funding"].sum()) if len(win) else 0.0


LAST_PICKS_FILE = "last_picks.json"


def _load_last_picks() -> dict:
    """上次交付的 {symbol: direction}，用于惰性再平衡标注（新进/持续/反转/该平）。"""
    p = mcfg.OUTPUT_DIR / LAST_PICKS_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_last_picks(picks: dict) -> None:
    mcfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (mcfg.OUTPUT_DIR / LAST_PICKS_FILE).write_text(json.dumps(picks, ensure_ascii=False), encoding="utf-8")


def _fuse_horizons(results: dict, horizons: list[int], tgt: str = "cs"):
    """多窗口融合 pred_z：共识强度 = 各窗口符号一致才强，打架就弱。

    基础 = 最长窗口的 cur（含 close/atr 等列），pred_z 换成「各窗口融合值」。
    融合值 = 各窗口 pred_z 均值 × 共识度（符号一致占比），不套复杂公式。
    tgt 指定融合哪个目标（cs / mn / va），供四套方案各自融合。
    """
    base = results.get((tgt, horizons[-1]))
    if base is None:
        return None
    zcols = {}
    for h in horizons:
        cur = results.get((tgt, h))
        if cur is not None:
            zcols[h] = cur.set_index("symbol")["pred_z"]
    if not zcols:
        return base
    zdf = pd.DataFrame(zcols)
    signs = np.sign(zdf)
    agree = signs.sum(axis=1).abs() / signs.count(axis=1)   # 0..1，符号一致占比
    fused_z = zdf.mean(axis=1) * agree                       # 打架时压缩强度
    out = base.copy()
    out["pred_z"] = out["symbol"].map(fused_z)
    return out.dropna(subset=["pred_z"])


def top_picks(cur, top_n=TOP_N):
    """按截面强度 |pred_z| 取最强 top_n 个，方向由 pred_z 符号决定。"""
    ranked = cur.reindex(cur["pred_z"].abs().sort_values(ascending=False).index)
    return ranked.head(top_n)


def build_card(cur, t_max, n_train, top_n=TOP_N, save_picks: bool = True):
    dt = datetime.fromtimestamp(int(pd.Timestamp(t_max).timestamp()),
                                tz=timezone.utc).strftime("%m-%d %H:%M UTC")
    top = top_picks(cur, top_n)
    last = _load_last_picks()
    lines = [f"**今日 {len(top)} 个操作币**（明确价格，照着下单）"]
    new_picks = {}
    changed = False
    for i, (_, r) in enumerate(top.iterrows(), 1):
        direction = "做多" if r["pred_z"] > 0 else "做空"
        sgn = 1 if r["pred_z"] > 0 else -1
        px = float(r["close"])
        atr = max(float(_pct(r.get("atr14_norm"), 0.03) or 0.03), 0.01)
        stop_dist = ATR_MULT * atr                          # 止损距离 = 3×ATR，给足空间
        zabs = abs(float(r["pred_z"]))
        pos = min(0.10 + 0.05 * min(zabs, 2.0), 0.20)       # 仓位：10% 起步，|z| 越大越重，上限 20%
        stop = px * (1 - sgn * stop_dist)                   # 止损价
        reduce_at = px * (1 + sgn * 2 * atr)                # 减仓价：方向有利 2×ATR 减半锁利
        add_at = px * (1 - sgn * 1 * atr)                   # 加仓价：回踩 1×ATR 顺势加
        sym = r["symbol"]
        name = sym.replace("USDT", "")
        new_picks[sym] = 1 if sgn > 0 else -1
        if sym not in last:
            tag = "[新进]"
            changed = True
        elif last[sym] == new_picks[sym]:
            tag = "[持续]"
        else:
            tag = "[反转]"
            changed = True
        lines.append(f"{i}. **{name}** {direction}｜仓位 {pos:.0%}｜{tag}")
        lines.append(f"   开 {_fmt_px(px)}｜止损 {_fmt_px(stop)}（{-sgn * stop_dist:+.0%}）｜"
                     f"减仓 {_fmt_px(reduce_at)}｜加仓 {_fmt_px(add_at)}")
        lines.append(f"   理由 {_reason(r)}")
    closed = [s.replace("USDT", "") for s in last if s not in new_picks]
    if closed:
        changed = True
        lines.append(f"**该平仓**：{'、'.join(closed)}")
    lines.append("")
    lines.append("**规则**：开仓后方向有利 +2×ATR 减半锁利，剩余移动止损 2×ATR 让利润跑，信号反转全平；止损 3×ATR 给足空间。")
    lines.append("**调仓**：信号变化才推送，只动「新进/反转/该平」，「持续」不动。")
    if save_picks:
        _save_last_picks(new_picks)
    return dt, lines, changed


# ============================================================ 四套方案（动作 C+D）
def _fetch_quote_vols() -> dict:
    """拉全市场 24h 成交额（quoteVolume），用于核心/卫星分层。失败返回空 dict。"""
    try:
        from model.data import _get_json
        tick = _get_json("/fapi/v1/ticker/24hr")
        return {d["symbol"]: float(d.get("quoteVolume", 0)) for d in tick}
    except Exception:
        return {}


def _pos_from_atr(r) -> float:
    """风险预算反推仓位：仓位 = 风险预算 ÷ (止损倍数 × ATR)，波动大的币自动小仓。

    = RISK_PER_TRADE / (ATR_MULT × atr14_norm)，clip 到 [2%, 25%]。
    这是动作 C 的核心：仓位随波动率调整，替代固定比例或「|z| 越大越重」的启发式规则。
    """
    atr = max(float(_pct(r.get("atr14_norm"), 0.03) or 0.03), 0.005)
    pos = RISK_PER_TRADE / (ATR_MULT * atr)
    return min(max(pos, 0.02), 0.25)


def _plan_lines(cur, title: str, only_long: bool = False, pos_scale: float = 1.0,
                top_n: int = TOP_N, trend: float = 0.0) -> list[str]:
    """把一组信号（cur：symbol/pred_z/close/atr）格式化成一套可下注清单。"""
    lines = [f"**{title}**"]
    if cur is None or cur.empty:
        lines.append("  （无信号）")
        return lines
    ranked = cur.reindex(cur["pred_z"].abs().sort_values(ascending=False).index)
    n = 0
    for _, r in ranked.iterrows():
        sgn = 1 if r["pred_z"] > 0 else -1
        if only_long and sgn < 0:          # 卫星腿：只做多，不做空
            continue
        # 弱信号过滤：|z| < Z 不做（与对账口径一致）
        if abs(float(r["pred_z"])) < Z:
            continue
        # 动作 E（趋势门控）：净敞口偏置——不关方向腿，只调多空权重
        tscale = _trend_scale(trend, sgn)
        # 动作 A（通用）：单边暴涨币禁止做空（|24h涨跌|≥20%，插针/资金费/下架）。做多不限。
        _ret24 = abs(float(np.expm1(_pct(r.get("ret_24h"), 0.0) or 0.0)))
        if sgn < 0 and _ret24 >= SHORT_MAX_RET24:
            continue
        px = float(r["close"])
        atr = max(float(_pct(r.get("atr14_norm"), 0.03) or 0.03), 0.01)
        stop_dist = ATR_MULT * atr
        pos = min(max(_pos_from_atr(r) * pos_scale * tscale, 0.02), 0.25)
        stop = px * (1 - sgn * stop_dist)
        sym = r["symbol"]
        name = sym.replace("USDT", "")
        direction = "做多" if sgn > 0 else "做空"
        n += 1
        lines.append(f"{n}. **{name}** {direction}｜仓位 {pos:.0%}")
        lines.append(f"   开 {_fmt_px(px)}｜止损 {_fmt_px(stop)}（{-sgn * stop_dist:+.0%}）")
        lines.append(f"   理由 {_reason(r)}")
        if n >= top_n:
            break
    if n == 0:
        lines.append("  （无信号）")
    return lines


def build_four_plans(results: dict, t_max, top_n: int = TOP_N, save_picks: bool = True,
                     trend: float = 0.0) -> tuple[str, list[str], bool]:
    """输出可独立下注的方案（动作 C+E+F）。

    core 腿（大币 + cs 截面）已砍（2026-09-01）：cs/va 截面信号只在中小币有效，
    大币 beta 高度相关、截面排序区分度低（ACFR 实证加密截面动量弱），大币职能归 beta 腿。
    卫星/组合腿此前已砍。当前唯一截面腿：
    ① 中小币截面腿：va 标签（波动率调整），市值 top30% 以下 + 流动性门槛，多空。
    仓位走风险预算反推（动作 C）+ 趋势门控（动作 E）+ 波动率门控 + 入局时机位置过滤（动作 F）。
    """
    dt = datetime.fromtimestamp(int(pd.Timestamp(t_max).timestamp()),
                                tz=timezone.utc).strftime("%m-%d %H:%M UTC")

    va = _fuse_horizons(results, HORIZONS, "va")

    z = trend
    net = float(np.tanh(z))
    tilt = "偏多" if net > 0.05 else ("偏空" if net < -0.05 else "中性")
    lines = [f"**今日下注方案 · {dt}**｜趋势 z={z:+.2f}（净敞口 {net:+.0%}，{tilt}）",
             "净敞口=tanh(趋势z) 连续随行情强弱，方向腿不关死（多强空弱照常）", ""]

    # ① 中小币截面腿：va 标签，市值 top30% 以下 + 流动性门槛（截面排序只在中小币有效）
    all_syms = list(va["symbol"].unique()) if va is not None and not va.empty else []
    profile = md.market_profile(all_syms)
    large = md.large_cap_symbols(all_syms, profile)
    tradeable = md.tradeable_symbols(profile)
    mid_small = va[va["symbol"].map(lambda s: s not in large and s in tradeable)] if va is not None else None
    lines += _plan_lines(mid_small, "① 中小币截面腿（va 标签，多空）", top_n=top_n, trend=trend)
    lines.append("")

    # 事件驱动：方案①中小币截面腿的持仓集合变化才算 changed（决定是否推送，避免每小时刷屏）
    new_picks = {}
    if mid_small is not None:
        for _, r in mid_small.iterrows():
            sgn = 1 if r["pred_z"] > 0 else -1
            if abs(float(r["pred_z"])) < Z:
                continue
            new_picks[r["symbol"]] = sgn
    last = _load_last_picks()
    changed = set(new_picks.items()) != set(last.items())
    if save_picks and new_picks:
        _save_last_picks(new_picks)
    return dt, lines, changed


def four_plans_holdings(results: dict, t_max, top_n: int = TOP_N, trend: float = 0.0,
                        vol_z: float = 0.0) -> dict:
    """四套方案的结构化持仓（供纸面开仓 / 追踪净值用）。

    返回 {plan: {symbol: {"side": ±1, "pos": 仓位%, "price": 开仓价, "atr": atr_norm, "stop": 止损价,
                         "z": 截面z强度, "bb_pctb": 布林带位置, "sma20": 距20均线, "d_high48"/"d_low48": 距48h高低点,
                         "rsi": RSI, "ret24": 24h涨跌}}}。
    与 build_four_plans 同口径：核心/卫星/赔率修正/组合，仓位走风险预算反推 + 趋势门控。
    """
    va = _fuse_horizons(results, HORIZONS, "va")

    # 分层体系（连续，非二元标签）：规模=市值分位数，可交易性=流动性门槛，换手率=连续特征
    all_syms = list(va["symbol"].unique()) if va is not None and not va.empty else []
    profile = md.market_profile(all_syms)

    def _detail(r) -> dict:
        """落盘明细字段：给看板展示用（开价/布林带/通道/z位置），不影响下注逻辑。"""
        return {
            "z": round(float(r["pred_z"]), 4),
            "bb_pctb": round(float(_pct(r.get("bb_pctb"), 0.5) or 0.5), 4),
            "bb_width": round(float(_pct(r.get("bb_width"), 0.0) or 0.0), 4),
            "sma20": round(float(_pct(r.get("close_sma20"), 0.0) or 0.0), 4),
            "sma50": round(float(_pct(r.get("close_sma50"), 0.0) or 0.0), 4),
            "d_high48": round(float(_pct(r.get("dist_high48"), 0.0) or 0.0), 4),
            "d_low48": round(float(_pct(r.get("dist_low48"), 0.0) or 0.0), 4),
            "rsi": round(float(_pct(r.get("rsi14"), 50.0) or 50.0), 2),
            "ret24": round(float(np.expm1(_pct(r.get("ret_24h"), 0.0) or 0.0)), 4),
        }

    def _hold(cur, only_long: bool = False, pos_scale: float = 1.0) -> dict:
        out: dict = {}
        if cur is None or cur.empty:
            return out
        ranked = cur.reindex(cur["pred_z"].abs().sort_values(ascending=False).index)
        n = 0
        for _, r in ranked.iterrows():
            sgn = 1 if r["pred_z"] > 0 else -1
            if only_long and sgn < 0:
                continue
            if abs(float(r["pred_z"])) < Z:                 # 弱信号过滤
                continue
            tscale = _trend_scale(trend, sgn)               # 动作 E：净敞口偏置（不关腿）
            _ret24 = abs(float(np.expm1(_pct(r.get("ret_24h"), 0.0) or 0.0)))
            if sgn < 0 and _ret24 >= SHORT_MAX_RET24:      # 动作 A：单边暴涨币禁做空
                continue
            # 动作 F：入局时机位置过滤（布林带 %b）——方向不变，位置走过头就轻仓
            w_pos = _pos_weight(sgn, _pct(r.get("bb_pctb"), 0.5) or 0.5)
            # 流动性缩放（连续）：低换手率小仓、高换手率可重仓，替代旧的「成交额硬切大小」
            liq_scale = md.liquidity_scale(profile.get(r["symbol"], {}).get("turnover", 0.0))
            # 波动率门控（连续）：高波动缩总仓，极端波动接近离桌
            vscale = _vol_scale(vol_z)
            pos = min(max(_pos_from_atr(r) * pos_scale * tscale * w_pos * liq_scale * vscale, 0.02), 0.25)
            _atr = float(_pct(r.get("atr14_norm"), 0.03) or 0.03)
            _px = float(r["close"])
            # 限价入场：做多等回调 1×ATR 再买、做空等反弹 1×ATR 再空（不保证立即成交，甚至不成交）
            _limit = round(_px * (1.0 - sgn * LIMIT_ATR_MULT * _atr), 8)
            out[r["symbol"]] = {
                "side": sgn, "pos": round(pos, 4),
                "price": _px,                       # 信号参考价（展示用）
                "limit": _limit,                    # 限价成交价（结算用）
                "atr": _atr,
                "stop": round(_stop_price(_limit, sgn, _atr), 8),   # 止损基于限价成交价
                "w_pos": round(w_pos, 4),
                **_detail(r),
            }
            n += 1
            if n >= top_n:
                break
        return out

    # 规模=市值分位数 top 30%，可交易=流动性门槛（成交额+换手率），两层各干各的
    large = md.large_cap_symbols(all_syms, profile)
    tradeable = md.tradeable_symbols(profile)

    # cs/va 截面信号只在中小币有效（大币 beta 高度相关，截面排序区分度低，ACFR 实证加密截面动量弱）。
    # 大币职能归 beta 腿（趋势/动量），core 腿（大币+cs）错配已砍。
    # 卫星腿已砍（2026-09-01）：小币做多靠动量不靠截面 z。
    mid_small = va[va["symbol"].map(lambda s: s not in large and s in tradeable)] if va is not None else None
    return {
        "va": _hold(mid_small),      # 中小币截面腿（原 va 标签，划到中小币——截面排序唯一有效的地方）
    }


async def _push(title, lines):
    """决策机器人独立钉钉通道（与 monitor 告警机器人分离），支持加签。"""
    import aiohttp
    import base64
    import hashlib
    import hmac
    import urllib.parse

    webhook = ncfg.DINGTALK_DECISION_WEBHOOK
    secret = ncfg.DINGTALK_DECISION_SECRET
    if not webhook:
        print("[decision] 决策钉钉 webhook 未配置，只打印不推送")
        return
    url = webhook
    if secret:
        ts = str(round(time.time() * 1000))
        sign_str = f"{ts}\n{secret}"
        digest = hmac.new(secret.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(digest))
        url = f"{url}&timestamp={ts}&sign={sign}"
    text = "### " + title + "\n" + "\n".join(f"> {ln}" for ln in lines)
    payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
    try:
        async with aiohttp.ClientSession(trust_env=True, proxy=ncfg.PROXY or None) as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                body = await resp.text()
                ok = resp.status == 200 and '"errcode":0' in body
                print(f"[decision] 推送{'成功' if ok else '失败'} HTTP {resp.status}: {body[:200]}")
    except Exception as exc:
        print(f"[decision] 推送异常: {exc}")


def run_once(a) -> None:
    """吞吐量流水线：增量数据 → 多目标×多周期预测全池 → 逐日结算到期 → 对账矩阵 → 交付清单 → 推送。"""
    pool = (mcfg.OUTPUT_DIR / POOL_FILE).read_text(encoding="utf-8").split()

    if not a.dry:
        print(f"[decision] 增量更新数据（池 {len(pool)} 币，只补缓存缺口）…")
        for i, sym in enumerate(pool, 1):
            try:
                mdata.fetch_klines(sym)
                mdata.fetch_funding(sym)
            except Exception as exc:
                print(f"  [{i}/{len(pool)}] {sym} 失败: {exc}")

    print("[decision] 构建特征面板…")
    panel = features.build_panel(progress=False, symbols=pool)
    feats = train.feature_cols(panel)

    results, t_max = decide_multi(panel, feats)
    if not results:
        print("[decision] 训练或当前截面数据不足，无法出决策")
        return

    db = MonitorDB(ncfg.DB_PATH)
    now = int(time.time())
    latest = panel[panel["open_time"] == t_max]

    # —— 逐日结算：遍历所有到期预测（多 target × 多 horizon，每条独立结算）——
    settled_n = 0
    for rowid, ts, sym, direction, ref_price, pred_z, horizon_h, target in db.open_predictions():
        if ts + horizon_h * 3600 > now:
            continue
        row = latest[latest["symbol"] == sym]
        if row.empty or not ref_price or ref_price <= 0:
            continue
        gross = float(row["close"].iloc[0]) / float(ref_price) - 1.0
        hit = 1 if (gross > 0) == (direction > 0) else 0
        # 净口径：方向×市场收益 − 双边成本 − 方向×资金费（做多付、做空收）
        ret = direction * gross - 2 * mcfg.COST_SIDE - direction * _funding_cost(sym, ts, horizon_h)
        db.settle_prediction(rowid, ret, hit)
        settled_n += 1

    def _snapshot(s: pd.Series) -> str:
        """特征快照：numpy/NaN -> JSON 可存，给未来监督学习重训当样本。"""
        out = {}
        for k, v in s.items():
            try:
                fv = float(v)
                out[k] = None if fv != fv else fv
            except (TypeError, ValueError):
                out[k] = str(v)
        return json.dumps(out, ensure_ascii=False)

    # —— 全池落表：多目标 × 多 horizon，每天落新预测（重叠=特性，滚动对账攒样本）——
    inserted_n = 0
    for (tgt_key, h), cur in results.items():
        for _, r in cur.iterrows():
            direction = 1 if r["pred_z"] > 0 else -1
            db.insert_prediction(r["symbol"], direction,
                                 float(r["close"]), float(r["pred_z"]), h,
                                 _snapshot(r[feats]), target=tgt_key)
            inserted_n += 1

    # —— 对账矩阵（多目标 × 多 horizon）——
    grid = db.prediction_report_grid(Z)
    report_lines = _build_report(grid, inserted_n, settled_n)

    # —— 自愈：只用主策略（cs × HORIZON）的前向真实期望调 TOP_N ——
    main_cell = grid.get("cs", {}).get(HORIZON, {"n": 0, "expectancy": 0.0})
    rep = {"n": main_cell["n"], "expectancy": main_cell["expectancy"]}
    st = _load_strategy()
    top_n, adjust_reason = _self_heal(st, rep)
    _save_strategy(st)

    # —— β 趋势跟随（主推）：纯 beta，方向跟随，不碰截面模型 ——
    beta_dir, beta_dev = _beta_position(panel)
    beta_hold = beta_holdings(panel, top_n=top_n)
    if beta_dir > 0:
        beta_lines = [f"**β 趋势跟随（主推）**：指数 > 30日均线 {beta_dev:+.1%} → 满仓做多全池（等权 {len(beta_hold)} 个）"]
        for s, v in list(beta_hold.items()):
            beta_lines.append(f"  · {s.replace('USDT', '')} 做多 {v['pos']:.0%} @ {_fmt_px(v['price'])}")
    else:
        beta_lines = [f"**β 趋势跟随（主推）**：指数 < 30日均线 {beta_dev:+.1%} → 空仓（躲下跌，等指数重回均线上方）"]

    # —— 交付四套方案（动作 C+D+E：核心/卫星/赔率修正/组合，挂趋势门控）——
    trend = _market_trend(panel)
    dt, lines, changed = build_four_plans(results, t_max, top_n, save_picks=not a.dry, trend=trend)

    final = list(report_lines)
    if adjust_reason:
        final.insert(len(report_lines), f"**自愈调整**：{adjust_reason}")
    final += [""] + beta_lines + [""] + lines

    print(f"\n=== 吞吐量对账 + 今日操作清单 · {dt} ===")
    for ln in final:
        print("  " + ln)

    if a.dry:
        print("\n[decision] --dry：仅打印，未推送")
        return

    # —— 事件驱动推送：信号有变化才推，无变化静默（落库照常）——
    if changed:
        asyncio.run(_push(f"今日对账 {dt}", final))
        print("\n[decision] 信号有变化，已推钉钉")
    else:
        print("\n[decision] 信号无变化（无新进/反转/该平），不推送，落库照常")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="决策告知机器人")
    p.add_argument("--dry", action="store_true", help="用现有缓存，只打印不推送")
    p.add_argument("--loop", action="store_true",
                   help=f"常驻：每 {RUN_INTERVAL_H}h 重算信号，只在信号变化时推送（事件驱动，不限定标准时间）")
    a = p.parse_args()

    if a.loop:
        while True:
            run_once(a)
            print(f"[decision] 下次检查 {RUN_INTERVAL_H}h 后（信号无变化则不推）")
            time.sleep(RUN_INTERVAL_H * 3600)
    else:
        run_once(a)


if __name__ == "__main__":
    main()