# -*- coding: utf-8 -*-
"""事件驱动腿：把 monitor 实时采集的事件流（events 表）翻译成可下注的纸面信号。

设计思路：
- 事件是「对市场被迫成交时刻」的快照，是否有 edge 由纸面净值验证。
- 方向语义明确的事件才进腿，方向含糊的不进入。
- 仓位小、止损宽、独立账户结算。

已排除（前三天全池大样本证伪）：
- funding 极值反向（极端后是动量不是反转）→ 不进腿
- 清算瀑布抄底（插针越深亏越狠）→ 不进腿

进腿的 5 类（方向语义清晰、symbol 可直接映射 USDT 永续）：
  1. large_trade  大单买入→多 / 大单卖出→空（看多/看空资金）
  2. flow         持续吸筹→多 / 持续派发→空（净主动买卖）
  3. oi_spike     价涨+OI增→多 / 价跌+OI减→空（动量微观确认）
                  价涨+OI减→空（空头回补，持续性存疑）
  4. depth        盘口失衡（方向从 detail 解析）
  5. funding      （已证伪，不生成信号，仅保留统计）

不进腿：whale_transfer（symbol 是链上格式 eth.USDC，无法映射到币永续标的）。
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional

from monitor import config as ncfg

# 事件腿只回看最近 N 小时的事件（太久的事件已过期，价值衰减）
LOOKBACK_HOURS = 24
# 事件腿单币最小仓位 / 最大仓位（小仓起步，先证后加）
EVENT_POS_MIN = 0.02
EVENT_POS_MAX = 0.10
# 事件信号冷却：同一 signal 键在 COOLDOWN 秒内不重复生成（避免同一事件反复刷）
EVENT_COOLDOWN_SEC = 6 * 3600


class EventSignal:
    """一条事件驱动下注信号。"""
    __slots__ = ("symbol", "side", "etype", "reason", "confidence", "ts")

    def __init__(self, symbol: str, side: int, etype: str, reason: str, confidence: float, ts: int):
        self.symbol = symbol          # 大写 USDT 永续 symbol
        self.side = side              # +1 多 / -1 空
        self.etype = etype            # 事件类型
        self.reason = reason          # 一句话理由（真实事件快照）
        self.confidence = confidence  # 0~1 置信度（同类事件数量加权）
        self.ts = ts                  # 事件时间戳


def _parse_direction(title: str, detail: str, etype: str) -> Optional[int]:
    """从事件标题/详情解析下注方向。返回 +1/-1/None(方向不明)。

    只认明确的措辞，方向含糊返回 None（不猜）。
    """
    t = title or ""
    try:
        j = json.loads(detail) if detail else {}
        lines = " ".join(j.get("lines", []))
    except Exception:
        lines = ""

    if etype == "large_trade":
        if "买入" in t or "看多" in t:
            return 1
        if "卖出" in t or "看空" in t:
            return -1
        return None

    if etype == "flow":
        if "吸筹" in t or "净主动买入" in t:
            return 1
        if "派发" in t or "净主动卖出" in t:
            return -1
        return None

    if etype == "oi_spike":
        # 解读行有明确的方向结论
        if "新多资金入场" in lines or "趋势可能延续" in lines:
            return 1
        if "多头止损离场" in lines:
            return -1
        if "空头回补推动" in lines:
            return -1           # 空头回补是脉冲，持续性存疑 → 偏空
        return None

    if etype == "depth":
        # 盘口失衡：bid 失衡偏多 / ask 失衡偏空
        if "买盘" in lines or "偏多" in t:
            return 1
        if "卖盘" in lines or "偏空" in t:
            return -1
        return None

    return None


def _parse_reason(detail: str) -> str:
    """从 detail 里抽出最有信息量的一句解读，作为下注理由。"""
    try:
        j = json.loads(detail) if detail else {}
        for ln in j.get("lines", []):
            if "解读" in ln:
                return ln.replace("解读:", "").replace("解读：", "").strip()
    except Exception:
        pass
    return ""


def _mappable(symbol: str) -> Optional[str]:
    """链上代币 symbol 映射到币永续。映射不了返回 None。"""
    s = (symbol or "").upper()
    # 直接是 USDT 永续格式（如 BTCUSDT）
    if s.endswith("USDT") and s != "USDTUSDT" and len(s) <= 20:
        return s
    return None


def fetch_event_signals(db_path: str | None = None) -> list[EventSignal]:
    """读 events 表最近 LOOKBACK_HOURS 小时，解析出可下注的事件信号。

    返回去重后的信号列表（同 symbol+etype+side 在冷却期内只取最新一条）。
    """
    db_path = db_path or ncfg.DB_PATH
    now = int(time.time())
    cutoff = now - LOOKBACK_HOURS * 3600

    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT ts, symbol, type, title, detail FROM events WHERE ts >= ? ORDER BY ts DESC",
            (cutoff,),
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"[event_leg] 读 events 失败: {exc}")
        return []

    # 去重分两级：
    #   1) 同 (symbol, etype, side) 冷却期内保最新一条（同一事件反复触发）
    #   2) 同 symbol 多空信号冲突时，保最新时间戳的方向（后到者覆盖先到者）
    by_type: dict[tuple[str, str, int], EventSignal] = {}
    for ts, symbol, etype, title, detail in rows:
        sym = _mappable(symbol)
        if sym is None:
            continue
        side = _parse_direction(title or "", detail or "", etype)
        if side is None:
            continue
        key = (sym, etype, side)
        if key in by_type and by_type[key].ts - ts < EVENT_COOLDOWN_SEC:
            continue
        by_type[key] = EventSignal(
            sym, side, etype, _parse_reason(detail or "") or title or etype, 1.0, ts)

    # 同 symbol 冲突消解：保留最新时间戳的那条方向
    by_sym: dict[str, EventSignal] = {}
    for sig in sorted(by_type.values(), key=lambda s: s.ts, reverse=True):
        if sig.symbol not in by_sym:
            by_sym[sig.symbol] = sig
        # 若同 symbol 已有信号且方向不同，比较时间戳，只保留最新的方向（后到覆盖）
        elif by_sym[sig.symbol].side != sig.side:
            if sig.ts > by_sym[sig.symbol].ts:
                by_sym[sig.symbol] = sig
 
    signals = list(by_sym.values())
    signals.sort(key=lambda s: s.ts, reverse=True)
    return signals


def event_holdings(signals: list[EventSignal], prices: dict, atr_map: dict | None = None) -> dict:
    """把事件信号转成结构化持仓（与 four_plans_holdings 同构）。

    {symbol: {"side": ±1, "pos": 仓位, "price": 现价, "atr": atr_norm}}
    仓位：事件腿一律小仓（EVENT_POS_MIN~MAX），不跑风险预算反推 —— 事件腿的本质是
    「快速积累证据」，不是「重仓博收益」，重仓留给被证明的腿。
    """
    out: dict = {}
    if not signals:
        return out
    from decision import _pct, _stop_price
    atr_map = atr_map or {}
    # 同 symbol 同 side 合并，不同 side 冲突时跳过后到的（保守）
    for sig in signals:
        sym = sig.symbol
        px = prices.get(sym)
        if px is None or px <= 0:
            continue
        if sym in out and out[sym]["side"] != sig.side:
            continue   # 方向冲突（多空信号打架），不进场
        _atr = float(atr_map.get(sym, 0.03) or 0.03)
        out[sym] = {
            "side": sig.side,
            "pos": EVENT_POS_MIN,   # 事件腿固定小仓起步
            "price": px,
            "atr": _atr,
            "stop": round(_stop_price(px, sig.side, _atr), 8),
        }
    return out


def event_lines(signals: list[EventSignal], prices: dict) -> list[str]:
    """事件腿的可读清单（进 decision / paper 推送）。"""
    from decision import _fmt_px
    if not signals:
        return ["**⑤ 事件驱动腿（微小仓，积累证据）**", "  （近 24h 无可下注事件）"]
    lines = [f"**⑤ 事件驱动腿（微小仓 {EVENT_POS_MIN:.0%}，先证后加）**",
             f"  近 24h 捕捉 {len(signals)} 条方向明确事件，映射到 {len(set(s.symbol for s in signals))} 个标的："]
    for sig in signals[:10]:
        px = prices.get(sig.symbol)
        pxs = f" @ {_fmt_px(px)}" if px else ""
        direction = "做多" if sig.side > 0 else "做空"
        lines.append(f"  · {sig.symbol.replace('USDT','')} {direction}{pxs}｜{sig.etype}: {sig.reason}")
    return lines


if __name__ == "__main__":
    sigs = fetch_event_signals()
    print(f"最近 {LOOKBACK_HOURS}h 解析出 {len(sigs)} 条事件信号：")
    for s in sigs:
        d = "做多" if s.side > 0 else "做空"
        print(f"  {s.symbol:12s} {d} | {s.etype:12s} | {s.reason}")
