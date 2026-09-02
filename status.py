# -*- coding: utf-8 -*-
"""全局状态看板：主从式布局 + 实时刷新 + 双主题。

布局（专业交易终端式）：
  · 顶部：标题 + 主题切换 + 刷新状态 + KPI 条
  · 左导航：7 腿列表（点击切换，高亮当前）
  · 右明细：选中腿的持仓表（币名/方向 sticky，指标列横向滚动）
  · 底部：预测成绩单 / 当前持仓（tab 切换）

用法（在项目根目录下）:
    python status.py              # 生成静态 status.html（不常驻）
    python status.py serve        # 起本地服务，浏览器打开 http://127.0.0.1:8765
                                  # 前端每 5s 刷新本地接口；服务端每 60s 才拉一次实时价（缓存，防 IP 封禁）

数据源：data/model_out/four_plans_state.json（7 腿持仓）+ data/monitor.db（持仓 + 预测结算）。
涨红跌绿（中国习惯）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "monitor.db"
STATE = ROOT / "data" / "model_out" / "four_plans_state.json"
OUT = ROOT / "data" / "model_out" / "status.html"

PLAN_NAMES = {
    "beta": "β 趋势跟随",
    "va": "中小币截面",
    "event": "事件驱动",
    "momentum": "慢动量v1·7天",
    "momentum_v2": "慢动量v2·30天",
}
# 每个腿空仓时的原因说明（不是错误，是当前截面无符合信号）
PLAN_EMPTY_REASON = {
    "beta": "当前空仓：指数 < 30日均线（熊市），β 腿躲下跌等重回均线上方",
    "va": "当前空仓：中小币池（市值 top30% 以下）无 |z|≥1 的 va 截面信号",
    "event": "当前空仓：近24h无可下注的方向明确事件",
    "momentum": "v1 已停用，等旧持仓到期结算后空仓",
    "momentum_v2": "当前空仓：趋势 z≤1.0（门控偏移），v2 腿空仓等趋势转强",
}
# 每个腿的决策规则（展示在表头，展示决策逻辑）
PLAN_RULES = {
    "beta": "规则：指数 vs 30日均线定方向 → 牛做多全池等权 / 熊空仓。纯 beta 跟随，不选币，市价立即成交。",
    "va": "规则：va标签(超额收益÷自身波动)排序中小币(市值top30%以下) → |z|≥1 出信号。截面排序只在中小币有效(大币beta高度相关区分度低)。限价入场：做多等回调1×ATR、做空等反弹1×ATR(不保证成交=不做)。仓位=风险预算÷(3×ATR)×趋势门控×波动率门控×布林带位置×换手率缩放。止损=3×ATR+移动止损。",
    "event": "规则：近24h事件(大单/吸筹/爆仓/OI异动)翻译方向 → 固定微小仓2%积累证据，市价立即成交（事件时效性强不挂单）。",
    "momentum": "v1 已停用：7 天回看样本外表现不佳，不再使用。等旧持仓到期结算后空仓。",
    "momentum_v2": "规则(2年回放+walk-forward验证的最优)：30天横截面动量(skip最近1天)排序全池 → 做多最强势top20 → 趋势门控 w=max(0,tanh(z−1.0))，z>1.0才开仓(砍假反弹)。每币5%等权满仓100%，止损3×ATR，持有96h。回测：累计+560%、回撤-43%、命中40%、Calmar12.9。只做多不做空弱势。",
}
TARGET_NAMES = {"cs": "截面z", "mn": "市场中性", "va": "波动率调整"}

# 实时价缓存（服务端全局，60s 刷新一次，绝不打爆 IP）
_price_cache: dict = {}
_price_cache_ts: float = 0.0
_PRICE_TTL = 60


def _fetch_prices(force: bool = False) -> dict:
    global _price_cache, _price_cache_ts
    now = time.time()
    if not force and _price_cache and (now - _price_cache_ts) < _PRICE_TTL:
        return _price_cache
    try:
        import requests
        proxies = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         proxies=proxies, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if r.status_code == 200:
            _price_cache = {d["symbol"]: float(d["markPrice"]) for d in r.json()}
            _price_cache_ts = now
    except Exception:
        pass
    return _price_cache


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _prediction_stats() -> dict:
    if not DB.exists():
        return {}
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT target, ret, hit FROM predictions WHERE settled=1")
    rows = cur.fetchall()
    conn.close()
    from collections import defaultdict
    g = defaultdict(list)
    for target, ret, _hit in rows:
        g[target].append(ret)
    out = {}
    for t, rs in g.items():
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        n = len(rs)
        wr = len(wins) / n if n else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        exp = wr * avg_win + (1 - wr) * avg_loss
        out[t] = {"n": n, "wr": wr, "avg_win": avg_win, "avg_loss": avg_loss,
                  "ratio": abs(avg_win / avg_loss) if avg_loss else float("inf"),
                  "exp": exp}
    return out


def _live_trades() -> list:
    if not DB.exists():
        return []
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT symbol, side, entry_price, leverage, size, stop_price, open_ts "
                "FROM live_trades WHERE status='open' ORDER BY open_ts")
    rows = cur.fetchall()
    conn.close()
    return rows


def _history_batches() -> dict:
    """读 decision_snapshots 表，按腿返回历史批次列表（每个批次 = 一次决策换仓）。

    返回 {leg: [{run_ts, trades: [{sym,side,pos,price,limit,stop,pnl,exit_price,hit_stop,filled,settled_ts}]}]}
    按 run_ts 倒序（最新批次在前）。解决「看板只显示当前持仓、历史批次被刷掉」的问题。
    """
    if not DB.exists():
        return {}
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decision_snapshots'")
    if not cur.fetchone():
        conn.close()
        return {}
    cur.execute("""
        SELECT run_ts, leg, symbol, side, pos, price, limit_px, stop,
               pnl, exit_price, hit_stop, filled, settled_ts
        FROM decision_snapshots ORDER BY run_ts DESC, leg, symbol
    """)
    rows = cur.fetchall()
    conn.close()

    out: dict = {}
    for run_ts, leg, symbol, side, pos, price, limit_px, stop, pnl, exit_price, hit_stop, filled, settled_ts in rows:
        batches = out.setdefault(leg, [])
        batch = None
        for b in batches:
            if b["run_ts"] == run_ts:
                batch = b
                break
        if batch is None:
            batch = {"run_ts": run_ts, "trades": []}
            batches.append(batch)
        batch["trades"].append({
            "sym": symbol, "side": side, "pos": pos, "price": price, "limit": limit_px,
            "stop": stop, "pnl": pnl, "exit_price": exit_price,
            "hit_stop": hit_stop, "filled": filled, "settled_ts": settled_ts,
        })
    return out


def _nav_series(state: dict) -> list:
    """从 history 提取「当前策略版本」的净值曲线段。

    history 列数随历史砍腿变化（5/6/7/8 列），且早期 4 列的列含义与现在不同。
    只取从最新往前、连续「列数 == len(PLAN_NAMES)」的记录（当前腿结构的连续段），
    遇到列数不同的记录（= 历史版本边界）就停止。返回 [{ts, navs:[...]}]。
    """
    hist = state.get("history", [])
    n = len(PLAN_NAMES)
    series = []
    for row in reversed(hist):
        if len(row) == n + 1:
            series.append({"ts": row[0], "navs": row[1:]})
        else:
            break          # 列数变了 = 策略版本边界，停止
    series.reverse()
    return series


def _replay_data() -> dict:
    """读历史回放结果（backtest/replay.py 落盘），供看板展示 2 年全历史多腿对比曲线。

    回放是「数据回放」不是「日历时间攒样本」——733 天 × 每天截面 × 未来 96h 结算，
    一次跑完 2 年。多腿同一市场环境平行对比，净值差异 = 纯策略差异。
    降采样到 ~220 点防前端卡顿。
    """
    p = Path("data/model_out/replay_all.json")
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        legs = d.get("legs", {})
        if not legs:
            return {}
        out = {"legs": {}, "n_points": 0}
        for name, res in legs.items():
            eq = res.get("equity", [])
            if not eq:
                continue
            out["n_points"] = max(out["n_points"], len(eq))
            step = max(1, len(eq) // 220)
            sampled = eq[::step]
            if len(sampled) > 1 and sampled[-1][0] != eq[-1][0]:
                sampled.append(eq[-1])
            out["legs"][name] = {"equity": sampled, "stats": res.get("stats", {})}
        return out
    except Exception:
        return {}


def _tracks_data() -> dict:
    """读独立轨道目录，返回所有跑过的独立轨道（任意时间段、与主状态无关）。"""
    tracks_dir = Path("data/model_out/tracks")
    if not tracks_dir.exists():
        return {"tracks": []}
    out = []
    for p in sorted(tracks_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            eq = d.get("equity", [])
            out.append({
                "name": d.get("name", p.stem),
                "start": d.get("start", ""), "end": d.get("end", ""),
                "stats": d.get("stats", {}),
                "equity": eq,
                "monthly": d.get("monthly", {}),
            })
        except Exception:
            continue
    return {"tracks": out}


def build_state_json(prices: dict, state: dict) -> dict:
    plans_out = {}
    for plan, name in PLAN_NAMES.items():
        p = state.get("plans", {}).get(plan, {})
        holds = p.get("holdings", {})
        longs = sum(1 for v in holds.values() if v.get("side", 1) > 0)
        shorts = sum(1 for v in holds.values() if v.get("side", 1) < 0)
        tot_pos = sum(v.get("pos", 0) for v in holds.values())
        items = []
        float_pnl = 0.0                        # 当前持仓的加权浮盈率（绝对仓位口径，随现价实时变）
        for sym, v in holds.items():
            side = v.get("side", 1)
            entry = v.get("price", 0.0)
            limit = v.get("limit")          # 限价成交价
            stop = v.get("stop")
            now = prices.get(sym, entry)
            # 成交状态与浮盈：限价腿(有 limit)用限价判断成交+算浮盈；市价腿(无 limit)立即成交用开仓价算
            if limit:
                filled = (side > 0 and now <= limit) or (side < 0 and now >= limit)
                pnl = side * (now / limit - 1.0) if filled else 0.0
            else:
                filled = True
                pnl = side * (now / entry - 1.0) if entry > 0 else 0.0
            float_pnl += pnl * v.get("pos", 0)   # 加权浮盈（pos 是占账户资金比例）
            hit_stop = False
            if stop and now:
                hit_stop = (side > 0 and now <= stop) or (side < 0 and now >= stop)
            items.append({
                "sym": sym, "side": side, "pos": v.get("pos", 0),
                "price": entry, "limit": limit, "stop": stop, "now": now,
                "pnl": pnl, "filled": filled,
                "hit_stop": hit_stop,
                "z": v.get("z"), "bb_pctb": v.get("bb_pctb"),
                "sma20": v.get("sma20"), "sma50": v.get("sma50"),
                "d_high48": v.get("d_high48"), "d_low48": v.get("d_low48"),
                "rsi": v.get("rsi"), "ret24": v.get("ret24"),
            })
        live_nav = p.get("nav", 1.0) * (1.0 + float_pnl)   # 实时净值 = 已实现 nav × (1+浮盈)
        plans_out[plan] = {
            "name": name, "nav": p.get("nav", 1.0), "ret": p.get("nav", 1.0) - 1.0,
            "live_nav": live_nav, "live_ret": live_nav - 1.0,
            "float_pnl": round(float_pnl, 6),
            "longs": longs, "shorts": shorts, "tot_pos": round(tot_pos, 4),
            "ts": p.get("ts"), "items": items,
            "empty_reason": PLAN_EMPTY_REASON.get(plan, "") if not items else "",
            "rules": PLAN_RULES.get(plan, ""),
        }

    stats = _prediction_stats()
    stats_out = {}
    for t in ["cs", "mn", "va"]:
        s = stats.get(t)
        if s:
            stats_out[t] = {"name": TARGET_NAMES.get(t, t), **s}
    ic_total = sum(s.get("n", 0) for s in stats_out.values())   # 已结算 IC 样本总数

    lives = []
    for sym, side, entry, lev, size, stop, open_ts in _live_trades():
        now = prices.get(sym, entry)
        pnl = side * (now / entry - 1.0) * lev
        lives.append({"sym": sym, "side": side, "entry": entry, "size": size,
                      "lev": lev, "stop": stop, "now": now, "pnl": pnl})

    # 持仓合计浮盈（等权口径，因为仓位都是 size 等权）
    live_total = sum(l["pnl"] for l in lives) / len(lives) if lives else None

    # 纸面净值区间（最好/最差腿）——用实时净值（已实现+浮盈），随现价变
    navs = [(PLAN_NAMES[k], p.get("live_nav", p.get("nav", 1.0))) for k, p in plans_out.items()]
    best = max(navs, key=lambda x: x[1]) if navs else None
    worst = min(navs, key=lambda x: x[1]) if navs else None

    return {
        "ts": time.time(),
        "price_n": len(prices),
        "started": state.get("started"),
        "trend": state.get("trend"),
        "plans": plans_out,
        "stats": stats_out,
        "ic_total": ic_total,
        "live": lives,
        "live_total": live_total,
        "history_batches": _history_batches(),
        "nav_series": _nav_series(state),
        "replay": _replay_data(),
        "tracks": _tracks_data(),
        "best_leg": {"name": best[0], "nav": best[1]} if best else None,
        "worst_leg": {"name": worst[0], "nav": worst[1]} if worst else None,
    }


def build_html(prices: dict) -> str:
    state = _load_state()
    data = build_state_json(prices, state)
    return _html_shell(json.dumps(data, ensure_ascii=False))


def _html_shell(initial_data_json: str) -> str:
    # 用普通字符串 + 占位符注入，JS 里 ${} 无需转义，彻底避免 f-string 转义地狱
    return _HTML.replace("__DATA__", initial_data_json)


_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>量子监控 · 全局状态</title>
<style>
:root{
  --bg:#070b14; --bg2:#0a1020; --panel:#0d1424; --panel2:#121a2e; --line:#1c2740;
  --tx:#dbe4f5; --mut:#6d7a99; --dim:#465270;
  --up:#f43f5e; --down:#10b981; --long:#f43f5e; --short:#10b981;
  --accent:#38bdf8; --stop:#f59e0b;
  --r-lg:3px; --r-md:2px;
  --shadow:none;
  --glow:0 0 0 1px rgba(56,189,248,.4),0 0 14px rgba(56,189,248,.18);
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
}
[data-theme="light"]{
  --bg:#f2f4f8; --bg2:#ffffff; --panel:#ffffff; --panel2:#f4f6fa; --line:#e0e4ee;
  --tx:#151a24; --mut:#6b7384; --dim:#aab2c0;
  --up:#e63946; --down:#0fbf7f; --long:#e63946; --short:#0fbf7f;
  --accent:#2563eb; --stop:#e8912d;
  --shadow:none;
  --glow:0 0 0 transparent;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);
  font:14px/1.55 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  overflow-y:auto;overflow-x:hidden;transition:background .2s,color .2s}
[data-theme="dark"] body{
  background-image:linear-gradient(rgba(56,189,248,.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(56,189,248,.025) 1px,transparent 1px);
  background-size:32px 32px}

/* ============ 顶栏（sticky 常驻） ============ */
.topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:14px;
  padding:12px 24px;border-bottom:1px solid var(--line);background:var(--bg2)}
.brand{font-size:16px;font-weight:700;letter-spacing:.2px}
.brand b{color:var(--accent);text-shadow:var(--glow)}
.topbar .sp{flex:1}
.status{font-size:12px;color:var(--mut);display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--down);animation:blink 1.8s infinite}
.dot.off{background:var(--dim);animation:none}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.theme-btn{background:var(--panel);border:1px solid var(--line);color:var(--tx);
  border-radius:var(--r-md);padding:5px 12px;font-size:12.5px;cursor:pointer;
  transition:background .15s,border-color .15s}
.theme-btn:hover{border-color:var(--accent)}

/* ============ KPI 条 ============ */
.kpis{display:flex;gap:12px;padding:14px 24px;flex-wrap:wrap}
.navchart{margin:0 24px 4px;background:var(--panel);border:1px solid var(--line);border-radius:var(--r-lg);box-shadow:var(--shadow)}
.navchart-h{padding:10px 16px 4px;font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.navchart-legend{font-weight:400;font-size:11.5px;display:flex;gap:10px}
.navchart-legend .lg{color:var(--mut)}
.navchart svg{display:block;padding:0 8px 8px}
.navchart-empty{padding:10px 16px;font-size:12px;color:var(--dim)}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-md);
  padding:12px 18px;min-width:170px;box-shadow:var(--shadow);flex:1}
.kpi .k{font-size:11px;color:var(--mut);margin-bottom:3px}
.kpi .v{font-size:22px;font-weight:700;font-family:var(--mono);font-variant-numeric:tabular-nums;letter-spacing:-.3px}
.kpi .v.sm{font-size:16px}
.kpi .sub{font-size:11px;color:var(--mut);margin-top:2px}
.kpi.trend-bull .v{color:var(--up)}
.kpi.trend-bear .v{color:var(--down)}
.kpi.trend-flat .v{color:var(--mut)}

/* ============ 横向腿 tab（终端式，锐利直角，不换行） ============ */
.legs{display:flex;gap:0;padding:10px 24px 0;border-bottom:1px solid var(--line);
  overflow-x:auto;scrollbar-width:none}
.legs::-webkit-scrollbar{display:none}
.leg-tab{flex-shrink:0;cursor:pointer;position:relative;
  display:flex;align-items:center;gap:10px;
  padding:9px 16px 11px;color:var(--mut);white-space:nowrap;
  border:1px solid transparent;border-bottom:none;border-radius:var(--r-md) var(--r-md) 0 0;
  margin-right:2px;transition:color .12s,background .12s,border-color .12s}
.leg-tab:hover{color:var(--tx);background:color-mix(in srgb,var(--accent) 6%,transparent)}
.leg-tab.active{color:var(--tx);background:var(--panel);
  border-color:var(--line);border-bottom:1px solid var(--panel);
  margin-bottom:-1px;box-shadow:var(--glow)}
.leg-tab .lt-nm{font-size:13px;font-weight:600}
.leg-tab .lt-nav{font-size:13px;font-weight:700;font-family:var(--mono);
  font-variant-numeric:tabular-nums}

/* ============ 明细表 ============ */
.detail{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-lg);
  overflow:hidden;box-shadow:var(--shadow);margin:12px 24px 0}
.detail-head{padding:14px 18px;border-bottom:1px solid var(--line);
  display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.detail-head .dn{font-size:16px;font-weight:700}
.detail-head .ds{font-size:12px;color:var(--mut)}
.detail-sub{padding:6px 18px 0;font-size:12px;color:var(--mut);border-bottom:1px solid var(--line);line-height:1.6}
.rules{padding:8px 18px;font-size:12px;color:var(--mut);background:color-mix(in srgb,var(--accent) 5%,transparent);
  border-bottom:1px solid var(--line);line-height:1.5}
.hist{padding:0 18px 14px;border-top:1px solid var(--line)}
.hist-h{padding:11px 0 8px;font-size:13px;font-weight:600}
.hist-wrap{overflow-x:auto}
.hist-wrap table{min-width:0}
.hist-wrap td,.hist-wrap th{font-size:12px}
.hist-empty{color:var(--dim);text-align:left;padding:14px 0;font-size:12.5px}
.table-wrap{overflow:auto;max-height:65vh;scrollbar-width:thin;scrollbar-color:var(--line) transparent}
.table-wrap::-webkit-scrollbar{height:10px;width:10px}
.table-wrap::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px;border:2px solid var(--panel)}
.table-wrap::-webkit-scrollbar-thumb:hover{background:var(--mut)}
.table-wrap::-webkit-scrollbar-track{background:transparent}
table{width:100%;border-collapse:separate;border-spacing:0;font-size:12.5px;min-width:1120px}
thead th{position:sticky;top:0;z-index:3;background:var(--panel2);color:var(--mut);
  font-weight:500;font-size:11.5px;padding:10px 12px;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--line)}
thead th.l,td.l{text-align:left}
tbody td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line);
  font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:hover td{background:color-mix(in srgb,var(--accent) 6%,transparent)}
td.sym{font-weight:600;text-align:left}
td.pin{position:sticky;left:0;z-index:2;background:var(--panel)}
th.pin{position:sticky;left:0;z-index:4;background:var(--panel2)}
tr:hover td.pin{background:var(--panel2)}
td.hitstop{box-shadow:inset 3px 0 0 var(--stop)}
.up,.long{color:var(--up)}
.down,.short{color:var(--down)}
.mut{color:var(--mut)}
.tag{display:inline-block;padding:1px 7px;border-radius:2px;font-size:11px;font-weight:600;line-height:1.5}
.tag.long{background:color-mix(in srgb,var(--up) 13%,transparent);color:var(--up)}
.tag.short{background:color-mix(in srgb,var(--down) 13%,transparent);color:var(--down)}
.tag.stop{background:color-mix(in srgb,var(--stop) 16%,transparent);color:var(--stop)}
.bb{position:relative;display:inline-block;width:86px;height:12px;background:var(--line);
  border-radius:2px;vertical-align:middle}
.bb i{position:absolute;left:0;top:0;bottom:0;border-radius:2px;
  background:linear-gradient(90deg,var(--down),#9aa1ad,var(--up))}
.bb em{position:absolute;left:0;right:0;top:0;bottom:0;text-align:center;font-style:normal;
  font-size:9.5px;line-height:12px;color:#fff;text-shadow:0 0 3px rgba(0,0,0,.85)}

/* ============ 底部双栏（预测成绩单 | 当前持仓） ============ */
.bottom{display:flex;gap:14px;padding:14px 24px 28px;align-items:stretch}
.bpanel{flex:1;min-width:0;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--r-lg);box-shadow:var(--shadow);overflow:hidden;
  display:flex;flex-direction:column}
.bpanel .bh{padding:11px 16px;font-size:13px;font-weight:600;border-bottom:1px solid var(--line);flex-shrink:0}
.bpanel .bd{flex:1;display:flex;flex-direction:column}
.bpanel table{min-width:0}
.bpanel thead th{position:static}
.bpanel td,.bpanel th{font-size:12px}
.empty{color:var(--dim);text-align:center;padding:20px;font-size:12.5px}
.note{color:var(--mut);font-size:11.5px;padding:9px 16px;margin-top:auto}

@media (max-width: 900px){
  .bottom{flex-direction:column}
}
</style></head><body data-theme="dark">

<div class="topbar">
  <span class="brand">量子<b>监控</b></span>
  <span class="sp"></span>
  <span class="status"><span class="dot" id="dot"></span><span id="statusTxt">连接中…</span></span>
  <button class="theme-btn" id="themeBtn" type="button">切换亮色</button>
</div>

<div class="kpis" id="kpis"></div>

<div class="navchart" id="replay"></div>

<div class="navchart" id="tracks"></div>

<div class="legs" id="legs"></div>
<div class="detail" id="detail"></div>

<div class="bottom">
  <div class="bpanel">
    <div class="bh">预测结算成绩单</div>
    <div id="stats"></div>
  </div>
  <div class="bpanel">
    <div class="bh">当前持仓</div>
    <div id="live"></div>
  </div>
</div>

<script>
let DATA = __DATA__;
let selected = null;
const API = "/api/state";

const pct = x => (x==null||isNaN(x)) ? "—" : (x>=0?"+":"")+(x*100).toFixed(2)+"%";
const px = x => {
  if(x==null) return "—";
  if(x>=1000) return x.toLocaleString("en-US",{maximumFractionDigits:0});
  const d = x>=1?2:(x>=0.1?4:6);
  return x.toLocaleString("en-US",{maximumFractionDigits:d});
};
const pnlCls = v => v>=0?"up":"down";
const sideTag = s => s>0 ? '<span class="tag long">多</span>' : '<span class="tag short">空</span>';
// 成交状态：触止损 > 挂单中(限价未成交) > 已成交
function statusTag(it){
  if(it.hit_stop) return '<span class="tag stop">触止损</span>';
  if(it.limit!=null && !it.filled) return '<span class="tag" style="background:color-mix(in srgb,var(--mut) 14%,transparent);color:var(--mut)">挂单中</span>';
  if(it.limit!=null && it.filled) return '<span class="tag" style="background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent)">已成交</span>';
  return '<span class="mut">—</span>';
}

function bbBar(b){
  if(b==null) return '<span class="mut">—</span>';
  const p = Math.max(0,Math.min(1,b))*100;
  return '<span class="bb"><i style="width:'+p.toFixed(0)+'%"></i><em>'+p.toFixed(0)+'%</em></span>';
}

function renderKpis(d){
  // 市场状态：趋势 z → 牛/熊/震荡
  const tr = d.trend;
  let trendHtml = '<div class="kpi"><div class="k">市场状态</div><div class="v sm">—</div><div class="sub">等待纸面引擎落盘趋势</div></div>';
  if(tr && tr.z != null){
    const z = tr.z;
    const cls = z > 0.5 ? "trend-bull" : (z < -0.5 ? "trend-bear" : "trend-flat");
    const label = z > 0.5 ? "牛市 · 偏多" : (z < -0.5 ? "熊市 · 偏空" : "震荡 · 中性");
    trendHtml = '<div class="kpi '+cls+'"><div class="k">市场状态</div><div class="v sm">'+label+'</div><div class="sub">趋势 z = '+z.toFixed(2)+'</div></div>';
  }
  // 持仓总浮盈
  const lt = d.live_total;
  const liveHtml = lt==null
    ? '<div class="kpi"><div class="k">持仓总浮盈</div><div class="v sm">无持仓</div></div>'
    : '<div class="kpi"><div class="k">持仓总浮盈（等权）</div><div class="v '+pnlCls(lt)+'">'+pct(lt)+'</div><div class="sub">'+((d.live||[]).length)+' 笔持仓</div></div>';
  // 纸面净值区间
  const b = d.best_leg, w = d.worst_leg;
  const navHtml = (b&&w)
    ? '<div class="kpi"><div class="k">纸面净值区间</div><div class="v sm"><span class="up">'+b.nav.toFixed(4)+'</span> ~ <span class="down">'+w.nav.toFixed(4)+'</span></div><div class="sub">最好 '+b.name+' · 最差 '+w.name+'</div></div>'
    : '<div class="kpi"><div class="k">纸面净值区间</div><div class="v sm">—</div></div>';
  // IC 样本（每天结算到期预测，判断信号有效性）
  const ic = d.ic_total || 0;
  const icHtml = '<div class="kpi"><div class="k">IC 样本</div><div class="v">'+ic+'</div><div class="sub">已结算预测 · 每天自动累积</div></div>';
  document.getElementById("kpis").innerHTML = icHtml + trendHtml + liveHtml + navHtml;
}

function renderLegs(d){
  const el = document.getElementById("legs");
  const plans = d.plans||{};
  const keys = Object.keys(plans);
  if(!selected || !plans[selected]) selected = keys[0];
  el.innerHTML = keys.map(k=>{
    const p = plans[k];
    const active = k===selected ? "active" : "";
    const isEmpty = (p.items||[]).length===0;
    const navHtml = isEmpty
      ? '<span class="lt-nav mut">空仓</span>'
      : '<span class="lt-nav '+pnlCls(p.live_ret)+'">'+pct(p.live_ret)+'</span>';
    return '<div class="leg-tab '+active+'" data-plan="'+k+'">'+
      '<span class="lt-nm">'+p.name+'</span>'+navHtml+
    '</div>';
  }).join("");
}

function renderDetail(d){
  const el = document.getElementById("detail");
  const p = (d.plans||{})[selected];
  if(!p){
    el.innerHTML = '<div class="empty" style="padding:40px">暂无数据</div>';
    return;
  }
  const items = p.items||[];
  const ts = p.ts ? new Date(p.ts*1000).toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"}) : "—";
  if(items.length===0){
    el.innerHTML =
      '<div class="detail-head">'+
        '<span class="dn">'+p.name+'</span>'+
        '<span class="'+pnlCls(p.live_ret)+'" style="font-weight:700;font-size:18px">'+pct(p.live_ret)+'</span>'+
        '<span class="ds">净值 '+p.live_nav.toFixed(4)+' · '+ts+'</span>'+
      '</div>'+
      '<div class="empty" style="padding:28px;text-align:left">'+p.empty_reason+'</div>';
    return;
  }
  const rows = items.map(it=>{
    const hit = it.hit_stop ? ' hitstop' : '';
    return '<tr class="'+hit+'">'+
      '<td class="sym pin">'+it.sym.replace("USDT","")+'</td>'+
      '<td class="l">'+sideTag(it.side)+'</td>'+
      '<td>'+pct(it.pos)+'</td>'+
      '<td>'+px(it.price)+'</td>'+
      '<td>'+((it.limit!=null&&it.limit!==undefined)?px(it.limit):"—")+'</td>'+
      '<td>'+px(it.now)+'</td>'+
      '<td class="'+pnlCls(it.pnl)+'">'+pct(it.pnl)+'</td>'+
      '<td class="'+((it.z||0)>=0?'up':'down')+'">'+(it.z!=null?it.z.toFixed(2):"—")+'</td>'+
      '<td>'+bbBar(it.bb_pctb)+'</td>'+
      '<td class="'+((it.sma20||0)>=0?'up':'down')+'">'+(it.sma20!=null?pct(it.sma20):"—")+'</td>'+
      '<td class="mut">'+(it.sma50!=null?pct(it.sma50):"—")+'</td>'+
      '<td>'+(it.d_high48!=null?pct(it.d_high48):"—")+'</td>'+
      '<td>'+(it.d_low48!=null?pct(it.d_low48):"—")+'</td>'+
      '<td class="'+(it.rsi>=70?'up':(it.rsi<=30?'down':''))+'">'+(it.rsi!=null?it.rsi.toFixed(0):"—")+'</td>'+
      '<td class="'+((it.ret24||0)>=0?'up':'down')+'">'+(it.ret24!=null?pct(it.ret24):"—")+'</td>'+
      '<td>'+px(it.stop)+'</td>'+
      '<td>'+statusTag(it)+'</td>'+
    '</tr>';
  }).join("");

  el.innerHTML =
    '<div class="detail-head">'+
      '<span class="dn">'+p.name+'</span>'+
      '<span class="'+pnlCls(p.live_ret)+'" style="font-weight:700;font-size:18px">'+pct(p.live_ret)+'</span>'+
      '<span class="ds">净值 '+p.live_nav.toFixed(4)+' · '+p.longs+'多'+p.shorts+'空 · 仓'+(p.tot_pos*100).toFixed(0)+'% · '+ts+'</span>'+
    '</div>'+
    '<div class="rules">'+p.rules+'</div>'+
    '<div class="table-wrap"><table>'+
      '<thead><tr>'+
        '<th class="l pin">币</th><th class="l">方向</th><th>仓位</th><th>开价</th><th>限价</th><th>平价</th><th>浮盈</th>'+
        '<th>z</th><th>布林带</th><th>距20均线</th><th>距50均线</th><th>距48h高</th><th>距48h低</th>'+
        '<th>RSI</th><th>24h涨跌</th><th>止损价</th><th>状态</th>'+
      '</tr></thead>'+
      '<tbody>'+rows+'</tbody>'+
    '</table></div>'+
    renderHistory(d, selected);
}

function renderHistory(d, leg){
  const batches = (d.history_batches||{})[leg]||[];
  if(batches.length===0) return '<div class="hist-empty">暂无历史批次（等 paper 换仓后落库）</div>';
  const rows = batches.map(b=>{
    const dt = new Date(b.run_ts*1000).toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"});
    const settled = b.trades.some(t=>t.settled_ts);
    const totalPnl = b.trades.reduce((a,t)=>a+(t.pnl||0),0);
    const filledN = b.trades.filter(t=>t.filled).length;
    const status = settled ? '已结算' : (filledN>0?'持有中':'挂单中');
    return '<tr>'+
      '<td class="l">'+dt+'</td>'+
      '<td>'+b.trades.length+'</td>'+
      '<td class="'+pnlCls(totalPnl)+'">'+pct(totalPnl)+'</td>'+
      '<td>'+status+'</td>'+
    '</tr>';
  }).join("");
  return '<div class="hist"><div class="hist-h">历史批次（每次换仓落库，不随新仓消失）</div>'+
    '<div class="hist-wrap"><table><thead><tr><th class="l">开仓时间</th><th>持仓数</th><th>批次收益</th><th>状态</th></tr></thead>'+
    '<tbody>'+rows+'</tbody></table></div></div>';
}

function renderStats(d){
  const rows = Object.values(d.stats||{}).map(s=>{
    const ratio = s.ratio==Infinity?"∞":s.ratio.toFixed(2);
    return '<tr><td class="sym">'+s.name+'</td><td>'+s.n+'</td><td>'+(s.wr*100).toFixed(1)+'%</td>'+
      '<td class="up">'+pct(s.avg_win)+'</td><td class="down">'+pct(s.avg_loss)+'</td>'+
      '<td>'+ratio+'</td><td class="'+pnlCls(s.exp)+'">'+pct(s.exp)+'/笔</td></tr>';
  }).join("");
  document.getElementById("stats").innerHTML =
    '<div class="bd">'+
    '<table><thead><tr><th class="l">目标</th><th>样本</th><th>命中率</th><th>均赢</th><th>均亏</th><th>盈亏比</th><th>期望</th></tr></thead>'+
    '<tbody>'+(rows||'<tr><td colspan="7" class="empty">暂无已结算预测</td></tr>')+'</tbody></table>'+
    '<div class="note">盈亏比 &lt; 1 = 期望为负；&gt; 1 = 期望为正。期望 = 胜率×均赢 + 败率×均亏。</div>'+
    '</div>';
}

function renderLive(d){
  const rows = (d.live||[]).map(l=>{
    return '<tr><td class="sym">'+l.sym.replace("USDT","")+'</td><td class="l">'+sideTag(l.side)+'</td>'+
      '<td>'+px(l.entry)+'</td><td>'+pct(l.size)+'</td><td>'+l.lev+'x</td>'+
      '<td>'+px(l.stop)+'</td><td>'+px(l.now)+'</td>'+
      '<td class="'+pnlCls(l.pnl)+'">'+pct(l.pnl)+'</td></tr>';
  }).join("");
  document.getElementById("live").innerHTML =
    '<div class="bd">'+
    '<table><thead><tr><th class="l">币</th><th class="l">方向</th><th>开仓价</th><th>仓位</th><th>杠杆</th><th>止损价</th><th>现价</th><th>浮盈(杠杆后)</th></tr></thead>'+
    '<tbody>'+(rows||'<tr><td colspan="8" class="empty">无持仓</td></tr>')+'</tbody></table>'+
    '</div>';
}

function renderReplay(d){
  const el = document.getElementById("replay");
  const r = d.replay||{};
  const legs = r.legs||{};
  const keys = Object.keys(legs);
  if(keys.length===0){
    el.innerHTML = '<div class="navchart-empty">历史回放：未生成。跑 <code>python backtest/replay.py --save</code> 一次性重放 2 年数据，产出多腿对比净值曲线 + 数千笔样本。</div>';
    return;
  }
  const NAMES = {beta:'β 纯趋势跟随(基准)', v1:'慢动量v1·7天(已证伪)', v2:'★ 慢动量v2·30天(最优)'};
  const COLORS = {beta:'#8a94a6', v1:'#9aa1ad', v2:'#e63946'};
  const W=640, H=200, L=46, R=12, T=14, B=24;
  // 统一 y 范围（所有腿）
  const allNavs = keys.flatMap(k => (legs[k].equity||[]).map(p=>p[1]));
  if(allNavs.length===0){ el.innerHTML=''; return; }
  const mn = Math.min(...allNavs), mx = Math.max(...allNavs);
  const pad = (mx-mn)*0.12 || 0.05;
  const lo = Math.max(0, mn-pad), hi = mx+pad;
  const first = legs[keys[0]].equity||[];
  const x = i => L + (first.length===1?0:(W-L-R)*i/(first.length-1));
  const y = v => T + (H-T-B)*(1-(v-lo)/(hi-lo));
  let paths = '';
  let legend = '';
  keys.forEach(k=>{
    const eq = legs[k].equity||[];
    const st = legs[k].stats||{};
    let dpath='';
    for(let i=0;i<eq.length;i++){
      dpath += (i===0?'M':'L')+x(i).toFixed(1)+','+y(eq[i][1]).toFixed(1);
    }
    paths += '<path d="'+dpath+'" fill="none" stroke="'+COLORS[k]+'" stroke-width="2"/>';
    const tot = st.total_ret!=null ? (st.total_ret*100).toFixed(0)+'%' : '—';
    const dd = st.max_drawdown!=null ? (st.max_drawdown*100).toFixed(0)+'%' : '—';
    const cls = (st.total_ret||0)>=0 ? 'up' : 'down';
    legend += '<span class="lg" style="color:'+COLORS[k]+'">● '+(NAMES[k]||k)+' <b class="'+cls+'">'+tot+'</b> / 回撤 '+dd+'</span>';
  });
  const y1 = y(1.0);
  if(y1>=T && y1<=H-B) paths += '<line x1="'+L+'" y1="'+y1.toFixed(1)+'" x2="'+(W-R)+'" y2="'+y1.toFixed(1)+'" stroke="#555" stroke-width="1" stroke-dasharray="3,3"/>';
  const fmt = t => new Date(t*1000).toISOString().slice(0,10);
  el.innerHTML =
    '<div class="navchart-h"><b>2 年全历史回放 · v1(7天) vs v2(30天) 版本对比</b><span class="navchart-legend">'+legend+'</span></div>'+
    '<svg viewBox="0 0 '+W+' '+H+'" width="100%" preserveAspectRatio="xMinYMid meet">'+paths+
      '<text x="'+L+'" y="'+(H-6)+'" font-size="10" fill="#888">'+fmt(first[0][0])+'</text>'+
      '<text x="'+(W-R-60)+'" y="'+(H-6)+'" font-size="10" fill="#888">'+fmt(first[first.length-1][0])+'</text>'+
    '</svg>';
}

function renderTracks(d){
  const el = document.getElementById("tracks");
  const tracks = (d.tracks&&d.tracks.tracks)||[];
  if(tracks.length===0){
    el.innerHTML = '<div class="navchart-empty">独立轨道：未跑过。命令行 <code>python -m backtest.replay --start YYYY-MM-DD --end YYYY-MM-DD --track 名称</code> 任意时间段跑一条独立轨道（与主状态无关）。</div>';
    return;
  }
  const W=640, H=140, L=46, R=12, T=14, B=22;
  // 每个轨道一行: 名称+时间段+收益回撤+迷你净值曲线
  let rows = '';
  tracks.forEach((tk,i)=>{
    const st = tk.stats||{};
    const tot = st.total_ret!=null ? (st.total_ret*100).toFixed(1)+'%' : '—';
    const dd = st.max_drawdown!=null ? (st.max_drawdown*100).toFixed(1)+'%' : '—';
    const cls = (st.total_ret||0)>=0 ? 'up' : 'down';
    const eq = tk.equity||[];
    // 迷你曲线
    let svg='';
    if(eq.length>1){
      const navs = eq.map(p=>p[1]);
      const mn=Math.min(...navs), mx=Math.max(...navs);
      const pad=(mx-mn)*0.1||0.01;
      const lo=Math.max(0,mn-pad), hi=mx+pad;
      const x = j => L + (W-L-R)*j/(eq.length-1);
      const y = v => T + (H-T-B)*(1-(v-lo)/(hi-lo));
      let path='';
      for(let j=0;j<eq.length;j++){
        path += (j===0?'M':'L')+x(j).toFixed(1)+','+y(eq[j][1]).toFixed(1);
      }
      svg = '<svg viewBox="0 0 '+W+' '+H+'" width="300" height="60" preserveAspectRatio="none"><path d="'+path+'" fill="none" stroke="'+(st.total_ret>=0?'#e63946':'#0fbf7f')+'" stroke-width="1.5"/></svg>';
    }
    rows += '<div style="display:flex;align-items:center;gap:16px;padding:8px 16px;border-bottom:1px solid var(--line)">'+
      '<div style="min-width:120px"><b>'+tk.name+'</b><div class="mut" style="font-size:11px">'+tk.start+' ~ '+tk.end+'</div></div>'+
      '<div style="min-width:120px;font-size:13px">累计 <b class="'+cls+'">'+tot+'</b><br><span class="mut" style="font-size:11px">回撤 '+dd+' · 命中 '+(st.hit_rate? (st.hit_rate*100).toFixed(0)+'%':'—')+'</span></div>'+
      '<div style="flex:1">'+svg+'</div>'+
    '</div>';
  });
  el.innerHTML = '<div class="navchart-h"><b>独立轨道（任意时间段，与主状态无关）</b></div>'+rows;
}

function render(d){
  renderKpis(d);
  renderReplay(d);
  renderTracks(d);
  renderLegs(d);
  renderDetail(d);
  renderStats(d);
  renderLive(d);
  const t = new Date().toLocaleTimeString("zh-CN");
  document.getElementById("statusTxt").textContent = "刷新 "+t+" · 实时价 "+d.price_n+" 个";
}

/* ============ 事件委托（不用 inline onclick，避免 CSP 拦截） ============ */
document.getElementById("legs").addEventListener("click", function(e){
  const item = e.target.closest(".leg-tab");
  if(item){ selected = item.dataset.plan; render(DATA); }
});
document.getElementById("themeBtn").addEventListener("click", function(){
  const body = document.body;
  const isDark = body.dataset.theme === "dark";
  body.dataset.theme = isDark ? "light" : "dark";
  this.textContent = isDark ? "切换亮色" : "切换暗色";
});

render(DATA);
setInterval(async function(){
  try{
    const r = await fetch(API, {cache:"no-store"});
    const d = await r.json();
    DATA = d;
    render(d);
    document.getElementById("dot").classList.remove("off");
  }catch(e){
    document.getElementById("statusTxt").textContent = "刷新失败：" + e;
    document.getElementById("dot").classList.add("off");
  }
}, 5000);
</script>
</body></html>"""


def serve(port: int) -> None:
    import http.server
    import socketserver

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/state":
                prices = _fetch_prices()
                state = _load_state()
                data = build_state_json(prices, state)
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            elif self.path in ("/", "/index.html"):
                prices = _fetch_prices()
                html = build_html(prices)
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"[status] 看板已启动: http://127.0.0.1:{port}")
        print(f"[status] 前端每 5s 刷新本地接口；实时价每 {_PRICE_TTL}s 拉一次（缓存防 IP 封禁）")
        print(f"[status] Ctrl+C 退出")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[status] 已停止")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="全局状态看板")
    p.add_argument("cmd", nargs="?", default="", choices=["", "serve"], help="serve=起本地实时服务")
    p.add_argument("--port", type=int, default=8765, help="服务端口（默认 8765）")
    p.add_argument("--no-fetch", action="store_true", help="不拉实时价（离线）")
    a = p.parse_args()

    if a.cmd == "serve":
        serve(a.port)
        return

    prices = {} if a.no_fetch else _fetch_prices(force=True)
    html = build_html(prices)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"[status] 静态面板已生成: {OUT}")
    print(f"[status] 提示: 用 `python status.py serve` 起实时服务，浏览器打开 http://127.0.0.1:{a.port}")
    print(f"[status] 实时价 {len(prices)} 个 ｜ 纸面腿 {len(_load_state().get('plans', {}))} 个 ｜ 持仓 {len(_live_trades())} 笔")
    stats = _prediction_stats()
    if stats:
        print("\n预测结算成绩（已结算）:")
        for t in ["cs", "mn", "va"]:
            s = stats.get(t)
            if s:
                print(f"  {TARGET_NAMES.get(t, t):6} n={s['n']:3} 胜率{s['wr']*100:5.1f}% "
                      f"盈亏比{s['ratio']:.2f} 期望{s['exp']*100:+.3f}%/笔")


if __name__ == "__main__":
    main()
