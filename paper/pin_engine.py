# -*- coding: utf-8 -*-
"""插针抄底纸面引擎（事件驱动，独立账户）。

把 backtest/pin_strategy.py 验证过的 edge 变成可跑的纸面下注系统：
  - 信号：过去 15min 跌超 -5%（急跌=对手方被迫成交/爆仓清算），逐分钟判断 + 上升沿触发
  - 动作：做多，持有 12h 后平仓。仓位深度加权：-5% 插针=5%，越深下注越重
    （w=(depth/0.05)^2，-12% 插针≈5.8×，封顶 6×）
  - 同币 12h 内不重复开仓（去重，与回测一致）
  - 成本单边 6bp

数据源：klines_1m 表（需由 fetch_1m.py --days 1 定期保鲜；否则拿的是最后回填的历史价）。

回测依据（2026-09-02，33 币全量 × 2 年，816 独立事件，扣 6bp 单边）：
  等权 5%：均值 +2.06% / 命中 63.5% / 赔率比 2.02 / Sharpe 2.16 / 回撤 -11.9%
  深度加权(平方)：Sharpe 2.38 / 回撤 -7.5% / 分年（24/25/26）年年改善
  与慢动量 v2 日收益相关 ≈ 0（完全正交）

用法（在项目根目录下）：
    python -m paper.pin_engine               # 跑一次：扫信号 + 结算到期 + 推净值
    python -m paper.pin_engine --loop        # 常驻：每 5 分钟跑一次
    python -m paper.pin_engine --reset       # 清空状态（仅在需要时使用）

状态落盘 data/model_out/pin_paper_state.json；净值历史只增不改。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

DB = Path("data/monitor.db")
STATE = Path("data/model_out/pin_paper_state.json")

PIN_TH = -0.05          # 插针阈值：15min 跌超 -5%
LOOKBACK = 15           # 过去 15min 收益
HOLD_MIN = 720          # 持有 12h
PIN_POS = 0.05          # 基准仓位（-5% 插针 = 5%，深插针按权重放大）
POS_WEIGHT_EXP = 2      # 深度加权指数：0=等权，2=平方（回测甜点，Sharpe 2.16→2.38）
POS_WEIGHT_CAP = 6.0    # 权重上限（-15% 插针 (3.0)²=9 → 封顶 6 倍）
COST_SIDE = 0.0006      # 单边成本 6bp
POLL_SEC = 60           # 常驻轮询间隔 1 分钟（对齐回测逐分钟触发，不漏快速反弹的插针）
MIN_ROWS = 1_000_000    # 只交易有完整 2 年 1m 数据的币（与回测一致）
FRESH_MAX = 300         # 新鲜度护栏：最新 K 线距今 ≤ 5 分钟才允许开/平仓，否则跳过（防保鲜失败用旧价成交）

FAPI = "https://fapi.binance.com"
PROXY = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
REFRESH_MIN = 20        # 每次保鲜拉最近 20 分钟 1m K 线（15min 回看 + 余量）
REFRESH_SLEEP = 0.12    # 保鲜请求间隔（限速安全）


def _pos_weight(depth: float) -> float:
    """深度加权：越深插针赔率越高，下注越重。

    depth = -ret15（正值，越大越深）。-5% 插针权重=1，平方关系 w=(depth/0.05)^2，
    封顶 POS_WEIGHT_CAP。回测依据（33 币全量，2026-09-02）：
    等权 Sharpe 2.16 → 平方 2.38，回撤 -11.9%→-7.5%，分年（24/25/26）年年改善。
    """
    if POS_WEIGHT_EXP == 0:
        return 1.0
    return min((depth / 0.05) ** POS_WEIGHT_EXP, POS_WEIGHT_CAP)


def _refresh_1m(symbols: list[str]) -> int:
    """拉每个币最近 ~90 分钟 1m K线 upsert 进 klines_1m，返回写入条数。

    让纸面引擎自包含：不用外部定时跑 fetch_1m.py，--loop 常驻时每轮先保鲜再扫信号。
    幂等（INSERT OR REPLACE），限流退避，失败单币跳过不炸整轮。
    """
    import requests
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - REFRESH_MIN * 60_000
    total = 0
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    conn = sqlite3.connect(DB)
    for s in symbols:
        try:
            r = sess.get(f"{FAPI}/fapi/v1/klines", params={
                "symbol": s, "interval": "1m",
                "startTime": start_ms, "limit": 1000,
            }, proxies=PROXY, timeout=15)
        except Exception as exc:
            print(f"[pin_engine] {s} 保鲜请求失败 {exc}")
            continue
        if r.status_code in (429, 418):
            print(f"[pin_engine] {s} 保鲜限流 {r.status_code}，退避 20s")
            time.sleep(20)
            continue
        if r.status_code != 200:
            continue
        data = r.json()
        if not isinstance(data, list) or not data:
            continue
        rows = [(s, int(d[0]) // 1000, float(d[2]), float(d[3]), float(d[4]), float(d[5]))
                for d in data]
        conn.executemany(
            "INSERT OR REPLACE INTO klines_1m (symbol, open_time, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?)", rows)
        conn.commit()
        total += len(rows)
        time.sleep(REFRESH_SLEEP)
    conn.close()
    return total


def _symbols() -> list[str]:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT symbol, COUNT(*) c FROM klines_1m GROUP BY symbol HAVING c >= ?",
        (MIN_ROWS,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _latest_15m(symbols: list[str]) -> dict[str, tuple[int, float, float]]:
    """每币取最新 15 分钟收盘：返回 {symbol: (latest_ts, ret15, latest_close)}。"""
    conn = sqlite3.connect(DB)
    out: dict[str, tuple[int, float, float]] = {}
    for s in symbols:
        rows = conn.execute(
            "SELECT open_time, close FROM klines_1m WHERE symbol=? "
            "ORDER BY open_time DESC LIMIT ?",
            (s, LOOKBACK + 1),
        ).fetchall()
        if len(rows) < LOOKBACK + 1:
            continue
        rows = rows[::-1]                       # 升序
        c0 = rows[0][1]
        c1 = rows[-1][1]
        if c0 > 0 and c1 > 0:
            out[s] = (rows[-1][0], c1 / c0 - 1.0, c1)
    conn.close()
    return out


def _blank() -> dict:
    return {"started": None, "nav": 1.0, "positions": {}, "history": []}


def _load() -> dict:
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
            st.setdefault("started", None)
            st.setdefault("nav", 1.0)
            st.setdefault("positions", {})
            st.setdefault("history", [])
            return st
        except Exception:
            pass
    return _blank()


def _save(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def run(refresh: bool = False) -> None:
    if refresh:
        syms = _symbols()
        if syms:
            n = _refresh_1m(syms)
            print(f"[pin_engine] 1m 保鲜：{len(syms)} 币 upsert {n} 条")
    symbols = _symbols()
    if not symbols:
        print("[pin_engine] 无完整 1m 数据的币，先跑 fetch_1m.py 回填")
        return
    snap = _latest_15m(symbols)
    if not snap:
        print("[pin_engine] 1m 数据为空/过期")
        return

    st = _load()
    now = int(time.time())
    if st["started"] is None:
        st["started"] = now

    opened = []
    closed = []

    # 1) 平到期仓（12h 到期，按最新价结算，扣双边成本）
    for s, pos in list(st["positions"].items()):
        if s not in snap:
            continue
        if now - snap[s][0] > FRESH_MAX:        # 新鲜度护栏：数据过期不平（避免用旧价假结算）
            continue
        if now - pos["entry_ts"] >= HOLD_MIN * 60:
            exit_px = snap[s][2]
            ret = exit_px / pos["entry_px"] - 1.0 - 2 * COST_SIDE
            w = pos.get("weight", 1.0)
            st["nav"] = st["nav"] * (1.0 + PIN_POS * w * ret)
            closed.append((s, pos["entry_px"], exit_px, ret, w))
            del st["positions"][s]

    # 2) 开新仓（-5% 上升沿：本次触发且当前无持仓，深度加权）
    for s, (ts, ret15, close) in snap.items():
        if now - ts > FRESH_MAX:                # 新鲜度护栏：旧价不触发开仓
            continue
        if ret15 <= PIN_TH and s not in st["positions"]:
            depth = -ret15
            w = _pos_weight(depth)
            st["positions"][s] = {"entry_ts": ts, "entry_px": close,
                                  "ret15": ret15, "depth": depth, "weight": w}
            opened.append((s, close, ret15, w))

    # 3) 净值历史（只增不改）
    st["history"].append([now, round(st["nav"], 6)])

    lines = ["**插针抄底纸面（独立账户）**"]
    lines.append(f"净值 {st['nav']:.4f}（累计 {st['nav'] - 1:+.2%}）｜持仓 {len(st['positions'])} 个｜"
                 f"监控 {len(symbols)} 币")
    if opened:
        lines.append("开仓：" + "、".join(f"{s} 15min {r*100:.1f}% @{p:.4f}×{w:.2f}" for s, p, r, w in opened))
    if closed:
        lines.append("平仓：" + "、".join(f"{s} {r*100:+.2f}%" for s, _, _, r, _ in closed))
    if not opened and not closed:
        lines.append("本轮无触发（无 -5% 急跌，无到期平仓）")

    _save(st)
    print("\n".join(lines))

    # 只在有真实动作（开仓/平仓）时推送，空轮不推（避免 --loop 每分钟刷屏）
    if opened or closed:
        try:
            from decision import _push
            asyncio.run(_push("插针抄底纸面", lines))
        except Exception:
            pass


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="插针抄底纸面引擎")
    p.add_argument("--reset", action="store_true", help="清空状态重新开始")
    p.add_argument("--loop", action="store_true", help=f"常驻：每 {POLL_SEC // 60} 分钟跑一次（先保鲜 1m 再扫信号）")
    a = p.parse_args()
    if a.reset:
        if STATE.exists():
            STATE.unlink()
        print("[pin_engine] 状态已清空（净值从 1.0 重开）")
        return
    if a.loop:
        while True:
            run(refresh=True)
            time.sleep(POLL_SEC)
        return
    run(refresh=True)


if __name__ == "__main__":
    main()
