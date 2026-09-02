# -*- coding: utf-8 -*-
"""纸面交易引擎（多腿独立账户版）：beta / 中小币截面 / 事件 / 慢动量 四腿独立结算净值。

把「信号」翻译成「可独立下注的腿」，每腿一个独立纸面账户：
  - 每次运行：用期间 K线结算上一期持仓（真止损+移动止损+限价单成交）→ 累加净值 → 换新仓
  - 各腿净值独立追踪，依据净值表现取舍

用法（在项目根目录下）:
    python -m paper            # 跑一次：结算旧仓 + 换新仓 + 推净值
    python -m paper --loop     # 常驻：每 8h 跑一次
    python -m paper --reset    # 清空状态，重新开始

状态落盘 data/model_out/four_plans_state.json；决策快照落 data/monitor.db 的 decision_snapshots 表。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from decision import _push, decide_multi, four_plans_holdings, _market_trend, _market_vol, beta_holdings
from decision.event_leg import fetch_event_signals, event_holdings
from decision.momentum_leg import momentum_holdings
from model import config as mcfg, features, train

STATE_PATH = mcfg.OUTPUT_DIR / "four_plans_state.json"
POOL_FILE = "probe_pool.txt"
LOOP_HOURS = 96            # 常驻间隔 = 持有期 96h：每次 loop 都到期结算换仓，不白跑
HOLD_HOURS = 96            # 持有期 = 预测 horizon（96h/4天）：信号预测 4 天就持有 4 天，到期才结算换仓
                           # 治「预测 96h 但 8h 就结算、信号未兑现就被平」的周期错位
ATR_MULT = 3.0            # 止损距离 = 3×ATR，与 decision 模块一致
DB_PATH = mcfg.OUTPUT_DIR.parent / "monitor.db"   # data/monitor.db（决策快照复盘库）

PLAN_NAMES = {
    "beta": "β趋势跟随·牛多熊空",
    "va": "中小币截面·多空",
    "event": "事件驱动腿·微小仓",
    "momentum": "慢动量v1·7天(停用中)",
    "momentum_v2": "慢动量v2·30天回看",
}


def _fetch_prices() -> dict:
    """拉全市场 mark price。失败返回空 dict。

    用 requests + 代理 + 节流 + 重试（对齐 model/data.py 的成功路径）。
    之前用 urllib 无 UA 无节流触发 Binance 418 IP 信誉封禁，导致所有腿拿不到价。
    """
    import requests
    proxies = {"http": mcfg.PROXY, "https": mcfg.PROXY} if mcfg.PROXY else None
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"{mcfg.BASE_URL}/fapi/v1/premiumIndex"
    for attempt in range(4):
        try:
            resp = requests.get(url, params={}, timeout=20, proxies=proxies, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return {d["symbol"]: float(d["markPrice"]) for d in data}
        except requests.RequestException:
            pass
        time.sleep(2.0 * (attempt + 1))
    print("[paper] 拉实时价失败（重试 4 次仍失败，可能 418 封禁或代理未开）")
    return {}


def _blank_state() -> dict:
    return {"started": None,
            "plans": {k: {"nav": 1.0, "holdings": {}, "ts": None} for k in PLAN_NAMES},
            "history": []}


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            # 兼容旧结构 / 补齐四套：旧 plan dict 可能缺 nav/holdings/ts，逐个补，不覆盖已有值
            plans = st.setdefault("plans", {})
            for k in PLAN_NAMES:
                plan = plans.setdefault(k, {})
                plan.setdefault("nav", 1.0)
                plan.setdefault("holdings", {})
                plan.setdefault("ts", None)
            st.setdefault("history", [])
            st.setdefault("started", None)
            return st
        except Exception:
            pass
    return _blank_state()


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _settle_pnl(holdings: dict, prices: dict, entry_ts: int | None = None,
                now_ts: int | None = None) -> tuple[float, dict]:
    """一套方案的持仓收益（绝对仓位口径），带「真止损 + 移动止损」。

    返回 (pnl, settlements)：pnl 是本腿总收益；settlements = {symbol: {exit_price, hit_stop, pnl}}
    供落库回填复盘。

    Σ side×(exit/entry−1)×pos。

    关键：pos 是「占该腿账户总资金的比例」，直接乘，**不做 pos/Σpos 归一化**。
    - 空仓部分（1−Σpos）收益为 0，nav 反映真实总资金，不被满仓化放大。
    - 各腿真实仓位不同（va 中小币截面可能只投 20%、momentum 可能 1.4 倍杠杆），
      净值各自真实反映其风险收益。

    - 初始止损价 = holdings 里存的 `stop`（开仓价 × (1 ∓ 3×ATR)）。
    - 用持仓期间的 1h K线（high/low）逐根判断是否触及止损：
        做多：期间最低价 ≤ 止损价 → 按止损价结算（认亏离场，不扛到现价）
        做空：期间最高价 ≥ 止损价 → 按止损价结算
    - 移动止损（trailing / Chandelier）：价格朝有利方向走，止损价跟着上移/下移，
        让利润奔跑，只在回撤 3×ATR 时离场。未触及止损 → 按现价结算。

    无 K线数据（entry_ts 缺失、symbol 无 K线）时退化为现价结算，不清历史、不炸对账。
    """
    if not holdings:
        return 0.0, {}
    pnl = 0.0
    settlements = {}
    for s, v in holdings.items():
        side = v.get("side", 1)
        entry = v.get("limit") or v.get("price", 0.0)   # 限价成交价（旧数据无 limit 则退化为信号价）
        pos = v.get("pos", 0.0)
        atr = v.get("atr", 0.03) or 0.03
        stop = v.get("stop")
        limit_price = v.get("limit")
        if entry <= 0 or pos <= 0:
            continue
        if stop is None:
            stop = entry * (1.0 - side * ATR_MULT * atr)   # 兼容旧 holdings 无 stop
        exit_price, hit_stop, filled = _resolve_exit(s, side, atr, stop, limit_price,
                                                     entry_ts, now_ts, prices)
        if not filled:
            # 限价单未成交：收益 0，资金空仓（不做也是一种决策）
            settlements[s] = {"exit_price": None, "hit_stop": False, "pnl": 0.0, "filled": False}
            continue
        if exit_price is None or exit_price <= 0:
            continue
        one = side * (exit_price / entry - 1.0)
        pnl += one * pos
        settlements[s] = {"exit_price": exit_price, "hit_stop": hit_stop, "pnl": one, "filled": True}
    return pnl, settlements


def _resolve_exit(symbol: str, side: int, atr: float, stop: float, limit_price: float | None,
                  entry_ts: int | None, now_ts: int | None, prices: dict) -> tuple[float | None, bool, bool]:
    """限价单成交 + 止损/移动止损 → 返回 (exit_price, hit_stop, filled)。

    filled=False：限价单未成交（价格没回到目标入场价），收益 0、资金空仓 —— 这就是
    「挂单价太高做空，永远没到」的机器表达：不做也是一种决策。

    filled=True：exit_price 是离场价，hit_stop 表示是否触止损离场。

    流程：
      1) 逐根 1h K线找「限价成交点」：
         做多：期间最低价 ≤ limit_price → 成交（回调到位）
         做空：期间最高价 ≥ limit_price → 成交（反弹到位）
         全程未触及 → 不成交（filled=False）。
      2) 成交后，从成交那根 K线起判断止损 / 移动止损（Chandelier）：
         做多：stop 单调上移 = max(stop, 高点×(1−3×ATR))，只在 low ≤ stop 离场
         做空：stop 单调下移 = min(stop, 低点×(1+3×ATR))，只在 high ≥ stop 离场
      3) 全程未触止损 → 现价离场。
    """
    if entry_ts is None or now_ts is None or now_ts <= entry_ts:
        return prices.get(symbol), False, True   # 无区间 → 视为立即成交，现价结算
    rows = _fetch_period_klines(symbol, entry_ts, now_ts)
    if not rows:
        return prices.get(symbol), False, True   # 无 K线 → 现价结算

    # 1) 限价成交点
    if limit_price is None:
        filled_i = 0                             # 旧数据无 limit → 视为立即成交
    else:
        filled_i = None
        for i, (_ot, high, low, _close) in enumerate(rows):
            if side > 0:
                if low <= limit_price:
                    filled_i = i
                    break
            else:
                if high >= limit_price:
                    filled_i = i
                    break
        if filled_i is None:
            return None, False, False            # 未成交：挂单没等到

    # 2) 成交后止损 / 移动止损
    cur_stop = stop
    for _ot, high, low, _close in rows[filled_i:]:
        if side > 0:
            if low <= cur_stop:
                return cur_stop, True, True      # 触止损
            trail = high * (1.0 - ATR_MULT * atr)
            cur_stop = max(cur_stop, trail)      # 创新高 → 止损上移锁利
        else:
            if high >= cur_stop:
                return cur_stop, True, True
            trail = low * (1.0 + ATR_MULT * atr)
            cur_stop = min(cur_stop, trail)      # 创新低 → 止损下移锁利
    return prices.get(symbol), False, True       # 成交后未触止损 → 现价离场


def _fetch_period_klines(symbol: str, start_ts: int, end_ts: int) -> list:
    """取 [start_ts, end_ts] 区间的 1h K线 (open_time, high, low, close)，按时间升序。"""
    import sqlite3
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT open_time, high, low, close FROM klines "
            "WHERE symbol=? AND open_time>=? AND open_time<=? ORDER BY open_time",
            (symbol, start_ts, end_ts),
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _snapshot_table() -> None:
    """建决策快照表（若不存在）。每次 decision 落库，结算后回填，成为复盘资料。"""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts INTEGER NOT NULL,     -- 决策/开仓时间戳
            leg TEXT NOT NULL,           -- 腿名：beta/va/event/momentum
            symbol TEXT NOT NULL,
            side INTEGER,                -- +1 多 / -1 空
            pos REAL,                    -- 仓位（占腿账户资金比例）
            price REAL,                  -- 信号参考价
            limit_px REAL,               -- 限价成交价（做多=回调1ATR，做空=反弹1ATR）
            atr REAL,
            stop REAL,                   -- 止损价（基于限价成交价）
            w_pos REAL,                  -- 入局时机位置过滤权重（布林带）
            z REAL,                      -- 截面 z
            bb_pctb REAL,                -- 布林带 %b 位置
            rsi REAL,
            ret24 REAL,
            exit_price REAL,             -- 结算回填：离场价（限价单未成交则 NULL）
            pnl REAL,                    -- 结算回填：单笔收益（方向已乘入，绝对仓位口径）
            hit_stop INTEGER,            -- 结算回填：是否触止损 0/1
            filled INTEGER,              -- 结算回填：限价单是否成交 0/1（0=未成交，收益0）
            settled_ts INTEGER           -- 结算回填：结算时间
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ds_run ON decision_snapshots(run_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ds_leg ON decision_snapshots(leg, settled_ts)")
    conn.commit()
    conn.close()


def _insert_snapshots(holdings: dict, run_ts: int) -> None:
    """本次决策的 5 腿持仓全部落库（开仓快照，结算字段留空）。"""
    import sqlite3
    _snapshot_table()
    conn = sqlite3.connect(str(DB_PATH))
    for leg, hs in holdings.items():
        for sym, v in hs.items():
            conn.execute(
                "INSERT INTO decision_snapshots "
                "(run_ts, leg, symbol, side, pos, price, limit_px, atr, stop, w_pos, z, bb_pctb, rsi, ret24) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_ts, leg, sym, v.get("side", 1), v.get("pos", 0.0), v.get("price", 0.0),
                 v.get("limit"), v.get("atr"), v.get("stop"), v.get("w_pos"), v.get("z"),
                 v.get("bb_pctb"), v.get("rsi"), v.get("ret24")),
            )
    conn.commit()
    conn.close()


def _settle_snapshots(leg: str, run_ts: int | None, settlements: dict, settled_ts: int) -> None:
    """回填上一期快照的结算结果（离场价/单笔收益/是否触止损/是否成交/结算时间）。"""
    if not settlements or run_ts is None:
        return
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    for sym, st in settlements.items():
        conn.execute(
            "UPDATE decision_snapshots SET exit_price=?, pnl=?, hit_stop=?, filled=?, settled_ts=? "
            "WHERE run_ts=? AND leg=? AND symbol=? AND settled_ts IS NULL",
            (st["exit_price"], st["pnl"], 1 if st["hit_stop"] else 0,
             1 if st.get("filled", True) else 0, settled_ts, run_ts, leg, sym),
        )
    conn.commit()
    conn.close()


def run() -> None:
    pool = (mcfg.OUTPUT_DIR / POOL_FILE).read_text(encoding="utf-8").split()
    print(f"[paper] 构建特征面板（{len(pool)} 币）…")
    panel = features.build_panel(progress=False, symbols=pool)
    feats = train.feature_cols(panel)

    print("[paper] 训练四套方案模型（cs/va × 24/48/96）…")
    results, t_max = decide_multi(panel, feats)
    if not results:
        print("[paper] 训练或当前截面数据不足")
        return

    trend = _market_trend(panel)
    vol_z = _market_vol(panel)                       # 波动率门控：高波动降总仓
    holdings = four_plans_holdings(results, t_max, trend=trend, vol_z=vol_z)
    holdings["beta"] = beta_holdings(panel)          # β 趋势跟随：牛多熊空，纯 beta
    prices = _fetch_prices()
    # 事件驱动腿：读 events 表近 24h 方向明确事件，微小仓积累证据（第 6 个独立账户）
    esigs = fetch_event_signals()
    holdings["event"] = event_holdings(esigs, prices)
    # 慢动量 v1（7 天参数）：已停用，不再生成新持仓，等旧持仓 96h 到期结算后空仓
    holdings["momentum"] = {}
    # 慢动量 v2（30 天回看 + top20 + 门控偏移 1.0）：独立轨道，从今天起用新参数跑
    holdings["momentum_v2"] = momentum_holdings(panel, trend=trend, vol_z=vol_z)
    state = _load_state()
    now = int(time.time())
    if state["started"] is None:
        state["started"] = now

    lines = ["**纸面净值**（独立账户，持有期 96h 到期结算）"]
    newly_opened = {}                              # 本次「到期换仓/空仓开新」的持仓，才落快照
    for plan, h in holdings.items():
        pl = state["plans"][plan]
        entry_ts = pl.get("ts")
        expired = entry_ts is None or (now - entry_ts) >= HOLD_HOURS * 3600
        if pl["holdings"] and expired:             # 到期：结算旧仓（止损/移动止损/限价）+ 换新仓
            pnl, settlements = _settle_pnl(pl["holdings"], prices,
                                           entry_ts=entry_ts, now_ts=now)
            pl["nav"] = pl["nav"] * (1.0 + pnl)    # 复利
            _settle_snapshots(plan, entry_ts, settlements, now)   # 回填到期那批快照
            pl["holdings"] = h
            pl["ts"] = now
            newly_opened[plan] = h
        elif not pl["holdings"]:                   # 空仓：直接开新仓
            pl["holdings"] = h
            pl["ts"] = now
            newly_opened[plan] = h
        # else：未到期，继续持有（不结算不换仓）——信号预测 4 天就持有满 4 天
        nav = pl["nav"]
        status = "持有中" if (pl["holdings"] and not expired) else "已换仓"
        lines.append(f"· {PLAN_NAMES[plan]}：净值 {nav:.4f}（累计 {nav - 1:+.2%}）｜{len(pl['holdings'])} 持仓｜{status}")

    _insert_snapshots(newly_opened, now)           # 仅换仓/新开的持仓落快照（复盘资料）

    state["history"].append([now] + [state["plans"][k]["nav"] for k in PLAN_NAMES])
    # 落盘市场状态（趋势 z + 方向 + 指数偏离），供看板顶部展示「一眼决策结论」
    state["trend"] = {"z": round(float(trend), 4), "ts": now}
    _save_state(state)

    print("\n" + "\n".join(lines))
    asyncio.run(_push("纸面四套方案 · 结算换仓", lines))
    print("[paper] 已推钉钉")


def _reset() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    print("[paper] 四套方案纸面状态已清空")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="四套方案纸面交易引擎")
    p.add_argument("--reset", action="store_true", help="清空状态重新开始")
    p.add_argument("--loop", action="store_true", help=f"常驻：每 {LOOP_HOURS}h 结算+换仓+推净值")
    a = p.parse_args()
    if a.reset:
        _reset()
        return
    if a.loop:
        while True:
            run()
            print(f"[paper] 下次运行 {LOOP_HOURS}h 后")
            time.sleep(LOOP_HOURS * 3600)
        return
    run()


if __name__ == "__main__":
    main()
