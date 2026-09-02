# -*- coding: utf-8 -*-
"""数据资产清单 + 样例导出。

扫描项目全部数据资产（monitor.db 表、model_cache 缓存、model_out 回测输出、
纸面状态/流水、event_cache），生成：
  1. data/asset_inventory/数据资产清单.csv    —— 中文清单（资产/位置/规模/字段/时间范围/说明）
  2. data/asset_inventory/samples/<id>.csv   —— 每个资产一份样例（前 N 行）

用法（在项目根目录下）:  python -m model.inventory
"""
from __future__ import annotations

import csv
import glob
import json
import os
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "monitor.db"
OUT = DATA / "asset_inventory"
SAMPLES = OUT / "samples"
N_SAMPLE = 5


def _ts(ts) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


# ---------------------------------------------------------------- 数据库表
def db_assets() -> list[dict]:
    con = sqlite3.connect(DB)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    assets = []
    for t in tables:
        cols = cur.execute(f'PRAGMA table_info("{t}")').fetchall()
        schema = ",".join(c[1] for c in cols)
        n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        # 时间范围 + 币种覆盖（有 ts / symbol 列时）
        tmin = tmax = ""
        if any(c[1] == "ts" for c in cols):
            tmin, tmax = cur.execute(f'SELECT MIN(ts), MAX(ts) FROM "{t}"').fetchone()
            tmin, tmax = _ts(tmin), _ts(tmax)
        nsym = ""
        if any(c[1] == "symbol" for c in cols):
            nsym = cur.execute(f'SELECT COUNT(DISTINCT symbol) FROM "{t}"').fetchone()[0]
        assets.append({
            "id": f"db_{t}", "category": "monitor.db (SQLite)", "location": f"monitor.db::{t}",
            "size": f"{n} 行", "schema": schema, "range": f"{tmin} ~ {tmax}",
            "coins": nsym, "desc": _table_desc(t),
        })
    con.close()
    return assets


def _table_desc(t: str) -> str:
    d = {
        "trades": "逐笔成交/大单（taker_buy 标识主动买卖）",
        "oi": "持仓量（oi_base 币本位 / notional 美元本位）",
        "ratios": "多空比（全局/top账户/taker 三口径）",
        "mark_prices": "标记价 + 资金费率（每 1h 采样）",
        "onchain_txs": "链上鲸鱼转账（ETH/BSC）",
        "flow": "资金流净流入（窗口聚合）",
        "events": "监控告警事件（大单/爆仓/异动）",
        "predictions": "决策机器人预测落库（可结算 hit）",
        "heartbeats": "进程心跳（存活哨兵）",
        "ob_imbalance": "盘口失衡（bid/ask qty）",
        "data_gaps": "数据断档记录（健康哨兵）",
        "data_quality": "数据质检明细（当前为空）",
    }
    return d.get(t, "")


# ---------------------------------------------------------------- model_cache 缓存
def cache_assets() -> list[dict]:
    C = DATA / "model_cache"
    groups = [
        ("cache_klines_365d", "klines_365d_*.csv", "1h K线（365 天）", "open_time",
         "OHLCV + 成交笔数 + taker 主动买卖，模型主特征来源"),
        ("cache_klines_730d", "klines_730d_*.csv", "1h K线（730 天，2年重拉）", "open_time",
         "2 年数据重拉，样本 64→128 步"),
        ("cache_funding_365d", "funding_365d_*.csv", "资金费率（365 天）", "funding_time",
         "持仓资金费成本（回测 as-of 近似）"),
        ("cache_funding_730d", "funding_730d_*.csv", "资金费率（730 天）", "funding_time",
         "2 年资金费"),
        ("cache_funding_legacy", "funding_*.csv", "资金费率（旧 18 币）", "funding_time",
         "早期 18 币池资金费缓存"),
        ("cache_micro_lsr", "micro_lsr_*.csv", "多空比（微结构）", "ts",
         "账户多空比，微结构因子（Binance 限 30 天历史）"),
        ("cache_micro_oi", "micro_oi_*.csv", "持仓量 OI（微结构）", "ts",
         "OI 微结构因子"),
        ("cache_micro_taker_bs", "micro_taker_bs_*.csv", "taker 买卖（微结构）", "ts",
         "主动买卖盘微结构因子"),
        ("cache_micro_top_lsr", "micro_top_lsr_*.csv", "Top 账户多空比（微结构）", "ts",
         "大户多空比微结构因子"),
    ]
    assets = []
    for aid, pat, name, tcol, desc in groups:
        files = sorted(C.glob(pat))
        if not files:
            continue
        total = sum(f.stat().st_size for f in files)
        coins = len(files)
        # 代表文件（优先 BTCUSDT）
        rep = next((f for f in files if "BTCUSDT" in f.name), files[0])
        df = pd.read_csv(rep, nrows=2)
        schema = ",".join(df.columns)
        tmin = tmax = ""
        try:
            last = pd.read_csv(rep, usecols=[tcol]).iloc[:, 0]
            tmin, tmax = _ts(last.iloc[0]), _ts(last.iloc[-1])
        except Exception:
            pass
        assets.append({
            "id": aid, "category": "model_cache (缓存)", "location": f"model_cache/{pat}",
            "size": f"{coins} 币 × {total/1024:.0f}KB", "schema": schema,
            "range": f"{tmin} ~ {tmax}", "coins": coins, "desc": desc,
        })
    return assets


# ---------------------------------------------------------------- model_out 回测输出
def out_assets() -> list[dict]:
    O = DATA / "model_out"
    specs = [
        ("out_equity_curve", "equity_curve.csv", "净值曲线（回测逐 step 收益）"),
        ("out_feature_importance", "feature_importance.csv", "因子重要性（旧，无泄漏基线）"),
        ("out_feature_importance_96h", "feature_importance_96h.csv", "因子重要性（96h 截面z，去泄漏后）"),
        ("out_focused_sweep", "focused_sweep.csv", "12 配置去泄漏重扫（focused_sweep）"),
        ("out_sweep_results", "sweep_results.csv", "sweep 结果（= DSR 选择偏差基准）"),
        ("out_market_pool_200", "market_pool_200.txt", "200 币动态市场池"),
        ("out_probe_pool", "probe_pool.txt", "50 币扩池探针池"),
    ]
    assets = []
    for aid, fn, desc in specs:
        p = O / fn
        if not p.exists():
            continue
        if fn.endswith(".csv"):
            df = pd.read_csv(p)
            schema = ",".join(df.columns)
            n = len(df)
            tmin = tmax = ""
            for tcol in ("open_time", "horizon_h", "ts"):
                if tcol in df.columns:
                    tmin, tmax = str(df[tcol].iloc[0]), str(df[tcol].iloc[-1])
                    break
            assets.append({
                "id": aid, "category": "model_out (回测输出)", "location": f"model_out/{fn}",
                "size": f"{n} 行", "schema": schema, "range": f"{tmin} ~ {tmax}",
                "coins": "", "desc": desc,
            })
        else:  # txt 池文件
            syms = p.read_text(encoding="utf-8").split()
            assets.append({
                "id": aid, "category": "model_out (回测输出)", "location": f"model_out/{fn}",
                "size": f"{len(syms)} 币", "schema": "symbol", "range": "", "coins": len(syms),
                "desc": desc,
            })
    return assets


# ---------------------------------------------------------------- 纸面 + data 顶层
def paper_assets() -> list[dict]:
    A = []
    ledger = DATA / "paper_ledger.csv"
    if ledger.exists():
        df = pd.read_csv(ledger)
        A.append({"id": "paper_ledger", "category": "纸面交易 (data/)",
                  "location": "data/paper_ledger.csv", "size": f"{len(df)} 行",
                  "schema": ",".join(df.columns),
                  "range": _ts(df.ts.min()) + " ~ " + _ts(df.ts.max()), "coins": "",
                  "desc": "纸面交易流水（init/mark/rebalance 每步净值）"})
    st = DATA / "paper_state.json"
    if st.exists():
        s = json.loads(st.read_text(encoding="utf-8"))
        npos = len(s.get("positions", {}))
        A.append({"id": "paper_state", "category": "纸面交易 (data/)",
                  "location": "data/paper_state.json", "size": f"{npos} 仓",
                  "schema": "positions,entry_prices,equity,history,last_signal",
                  "range": _ts(s.get("started", 0)), "coins": "",
                  "desc": "纸面当前状态（仓位/入口价/净值/信号快照）"})
    ul = DATA / "universe_long.json"
    if ul.exists():
        u = json.loads(ul.read_text(encoding="utf-8"))
        A.append({"id": "universe_long", "category": "纸面交易 (data/)",
                  "location": "data/universe_long.json", "size": f"{len(u)} 币",
                  "schema": "symbol[]", "range": "", "coins": len(u),
                  "desc": "长列表候选币池"})
    return A


# ---------------------------------------------------------------- event_cache
def event_assets() -> list[dict]:
    A = []
    E = DATA / "event_cache"
    kp = E / "klines180.pkl"
    if kp.exists():
        obj = pickle.load(open(kp, "rb"))
        if isinstance(obj, dict):
            n = len(obj)
            rep = next(iter(obj.values()))
            schema = ",".join(rep.columns) if hasattr(rep, "columns") else ""
            A.append({"id": "event_klines180", "category": "event_cache",
                      "location": f"data/event_cache/klines180.pkl", "size": f"{n} 币 × {kp.stat().st_size/1024:.0f}KB",
                      "schema": schema, "range": "", "coins": n,
                      "desc": "事件研究用 180 天 K线缓存（内存 dict）"})
    fe = E / "flush_events.pkl"
    if fe.exists():
        obj = pickle.load(open(fe, "rb"))
        A.append({"id": "event_flush_events", "category": "event_cache",
                  "location": "data/event_cache/flush_events.pkl",
                  "size": f"{fe.stat().st_size/1024:.0f}KB", "schema": str(type(obj).__name__),
                  "range": "", "coins": "", "desc": "爆仓潮事件缓存"})
    return A


# ---------------------------------------------------------------- sample 导出
def export_sample(aid: str, n: int = N_SAMPLE) -> Path:
    """按资产 id 导一份样例 CSV。"""
    dest = SAMPLES / f"{aid}.csv"
    if aid.startswith("db_"):
        t = aid[3:]
        con = sqlite3.connect(DB)
        df = pd.read_sql_query(f'SELECT * FROM "{t}" ORDER BY rowid DESC LIMIT {n}', con)
        con.close()
        df.to_csv(dest, index=False, encoding="utf-8")
    elif aid in ("out_equity_curve", "out_feature_importance", "out_feature_importance_96h",
                 "out_focused_sweep", "out_sweep_results"):
        fn = aid[4:] + ".csv"
        df = pd.read_csv(DATA / "model_out" / fn, nrows=n)
        df.to_csv(dest, index=False, encoding="utf-8")
    elif aid.startswith("cache_"):
        C = DATA / "model_cache"
        patmap = {
            "cache_klines_365d": "klines_365d_*.csv", "cache_klines_730d": "klines_730d_*.csv",
            "cache_funding_365d": "funding_365d_*.csv", "cache_funding_730d": "funding_730d_*.csv",
            "cache_funding_legacy": "funding_*.csv", "cache_micro_lsr": "micro_lsr_*.csv",
            "cache_micro_oi": "micro_oi_*.csv", "cache_micro_taker_bs": "micro_taker_bs_*.csv",
            "cache_micro_top_lsr": "micro_top_lsr_*.csv",
        }
        files = sorted(C.glob(patmap[aid]))
        rep = next((f for f in files if "BTCUSDT" in f.name), files[0])
        df = pd.read_csv(rep, nrows=n)
        df.to_csv(dest, index=False, encoding="utf-8")
    elif aid in ("out_market_pool_200", "out_probe_pool"):
        fn = aid[4:] + ".txt"
        syms = (DATA / "model_out" / fn).read_text(encoding="utf-8").split()
        pd.DataFrame({"symbol": syms[:n]}).to_csv(dest, index=False, encoding="utf-8")
    elif aid == "paper_ledger":
        pd.read_csv(DATA / "paper_ledger.csv", nrows=n).to_csv(dest, index=False, encoding="utf-8")
    elif aid == "paper_state":
        s = json.loads((DATA / "paper_state.json").read_text(encoding="utf-8"))
        rows = [{"field": "equity", "value": s.get("equity")},
                {"field": "last_rebalance", "value": _ts(s.get("last_rebalance"))},
                {"field": "started", "value": _ts(s.get("started"))}]
        rows += [{"field": f"position.{k}", "value": v} for k, v in s.get("positions", {}).items()]
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8")
    elif aid == "universe_long":
        u = json.loads((DATA / "universe_long.json").read_text(encoding="utf-8"))
        pd.DataFrame({"symbol": u[:n]}).to_csv(dest, index=False, encoding="utf-8")
    elif aid == "event_klines180":
        obj = pickle.load(open(DATA / "event_cache" / "klines180.pkl", "rb"))
        rep = next(iter(obj.values()))
        rep.head(n).to_csv(dest, index=False, encoding="utf-8")
    elif aid == "event_flush_events":
        obj = pickle.load(open(DATA / "event_cache" / "flush_events.pkl", "rb"))
        pd.DataFrame([{"type": type(obj).__name__, "len": len(obj) if hasattr(obj, "__len__") else ""}]).to_csv(
            dest, index=False, encoding="utf-8")
    return dest


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    SAMPLES.mkdir(parents=True, exist_ok=True)

    assets = db_assets() + cache_assets() + out_assets() + paper_assets() + event_assets()

    # 清单 CSV
    cols = ["id", "category", "location", "size", "schema", "range", "coins", "desc"]
    inv = pd.DataFrame(assets, columns=cols)
    inv.columns = ["资产ID", "大类", "位置", "规模", "字段", "时间范围(UTC)", "币种数", "说明"]
    inv_path = OUT / "数据资产清单.csv"
    inv.to_csv(inv_path, index=False, encoding="utf-8-sig")  # utf-8-sig 便于 Excel 打开

    # 样例
    for a in assets:
        try:
            export_sample(a["id"])
        except Exception as e:
            print(f"[skip] {a['id']}: {e}")

    # 也写一份清单样例（清单本身的 sample）
    inv.head(N_SAMPLE).to_csv(SAMPLES / "数据资产清单_样例.csv", index=False, encoding="utf-8-sig")

    print(f"[inventory] 数据资产 {len(assets)} 项")
    print(f"[inventory] 清单 → {inv_path}")
    print(f"[inventory] 样例 → {SAMPLES} （{len(list(SAMPLES.glob('*.csv')))} 个文件）")
    for a in assets:
        print(f"   {a['id']:30s} {a['category']:22s} {a['size']}")


if __name__ == "__main__":
    main()
