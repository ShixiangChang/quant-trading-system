# -*- coding: utf-8 -*-
"""流动性数据管线：建映射 + 采流动性 + 落库。

映射：Binance /sapi/v1/capital/config/getall（权威，CEX 币）+ DEX 币用
  「权威地址精确查 → 标记价交叉验证」两级去歧义（不再靠 search 选流动性最大）。
流动性快照：CEX 币 → Binance 24h quoteVolume；DEX 币 → DexScreener TVL。
落库：token_map + liquidity 两张表。

用法: python -m model.liquidity
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from pathlib import Path

import requests

from monitor import config as ncfg, secrets
from monitor.db import MonitorDB
from . import config as mcfg

BASE = "https://api.binance.com"
CACHE = Path(__file__).resolve().parent.parent / "data" / "binance_coins.json"

# 合约 base → Binance 现货 coin 名（只有 symbol 与现货名不同的才需要）
SPOT_NAME = {"1000PEPE": "PEPE"}

# 已人工确认权威地址的 DEX 币（search 歧义严重 → 用精确地址查，零歧义）
AUTHORITY = {
    "龙虾": "0xeccbb861c0dda7efd964010085488b69317e4444",
    "BTR": "0xfed13d0c40790220fbde712987079eda1ed75c51",
    "CYS": "0x0c69199c1562233640e0db5ce2c399a88eb507c7",
    "MAGMA": "0x9f854b3ad20f8161ec0886f15f4a1752bf75d22261556f14cc8d3a1c5d50e529::magma::MAGMA",
}

# 英文搜索词（中文 symbol 需要英文名才能在 DEX 搜到）
SEARCH_NAME = {"龙虾": "lobster"}

# 标记价交叉验证的 gap 阈值：|DEX价 - 标记价|/标记价 > 此值 → 存疑（可能定位到同名错币）
GAP_THRESHOLD = 0.03


def _proxies():
    return {"http": ncfg.PROXY, "https": ncfg.PROXY} if ncfg.PROXY else None


def load_binance_coins(force: bool = False) -> dict:
    """Binance 现货所有币的「链 + 合约地址」映射，带本地缓存（避免每次调 signed 接口）。"""
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    ts = int(time.time() * 1000)
    qs = f"timestamp={ts}"
    sig = hmac.new(secrets.BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE}/sapi/v1/capital/config/getall?{qs}&signature={sig}"
    r = requests.get(url, headers={"X-MBX-APIKEY": secrets.BINANCE_API_KEY},
                     timeout=20, proxies=_proxies())
    r.raise_for_status()
    coins = {}
    for d in r.json():
        coins[d.get("coin")] = [
            {"network": n.get("network"), "contract": n.get("contractAddress") or "",
             "is_default": bool(n.get("isDefault"))}
            for n in d.get("networkList", [])
        ]
    CACHE.write_text(json.dumps(coins, ensure_ascii=False), encoding="utf-8")
    return coins


def dex_lookup(query: str) -> dict | None:
    """DexScreener search，取流动性最大的 pair。"""
    try:
        r = requests.get("https://api.dexscreener.com/latest/dex/search",
                         params={"q": query}, timeout=20)
        pairs = r.json().get("pairs", [])
    except Exception:
        return None
    if not pairs:
        return None
    best = max(pairs, key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0)
    return {
        "chain": best.get("chainId"),
        "address": (best.get("baseToken") or {}).get("address"),
        "liquidity": (best.get("liquidity") or {}).get("usd", 0) or 0,
        "volume": (best.get("volume") or {}).get("h24", 0) or 0,
    }


def fetch_mark_price(symbol: str) -> float | None:
    """Binance 合约标记价（fapi premiumIndex）。symbol 就是合约 symbol（含中文，如「龙虾USDT」）。"""
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": symbol}, timeout=15, proxies=_proxies())
        d = r.json()
        mp = float(d.get("markPrice") or 0)
        return mp if mp > 0 else None
    except Exception:
        return None


def dex_lookup_by_mark(query: str, mark_price: float | None) -> dict | None:
    """DexScreener search + 标记价交叉验证去歧义。

    同名 token 常跨多条链（wrapped/山寨），靠「流动性最大」会选错链。
    正解：每个链保留流动性最大的 pair 作候选，再选「价格最接近标记价」的那条链。
    标记价拿不到时退化为「流动性最大」。
    """
    try:
        r = requests.get("https://api.dexscreener.com/latest/dex/search",
                         params={"q": query}, timeout=20)
        pairs = r.json().get("pairs", [])
    except Exception:
        return None
    if not pairs:
        return None
    # 每个链只保留流动性最大的 pair（同名 token 跨链去重）
    per_chain: dict[str, dict] = {}
    for p in pairs:
        chain = p.get("chainId")
        liq = p.get("liquidity", {}).get("usd", 0) or 0
        px = float(p.get("priceUsd") or 0) or 0.0
        if chain not in per_chain or liq > per_chain[chain]["liquidity"]:
            per_chain[chain] = {
                "chain": chain,
                "address": (p.get("baseToken") or {}).get("address"),
                "liquidity": liq,
                "volume": (p.get("volume") or {}).get("h24", 0) or 0,
                "price": px,
            }
    if mark_price and mark_price > 0:
        priced = [c for c in per_chain.values() if c["price"] > 0]
        if priced:
            best = min(priced, key=lambda c: abs(c["price"] - mark_price))
            best["price_gap"] = abs(best["price"] - mark_price) / mark_price
            return best
    return max(per_chain.values(), key=lambda c: c["liquidity"])


def build_map(pool: list[str], coins: dict) -> list[dict]:
    """对 pool 每个币建映射：Binance 现货有 → cex；否则 DEX（权威地址优先 → 标记价交叉验证兜底）。"""
    rows = []
    for sym in pool:
        raw = sym[:-4] if sym.endswith("USDT") else sym
        spot = SPOT_NAME.get(raw, raw)
        if spot in coins:
            rows.append({"symbol": sym, "base": spot, "kind": "cex",
                         "chain": None, "contract_address": None, "source": "binance"})
        else:
            info = None
            mode = None
            if raw in AUTHORITY:                       # 1) 权威地址精确查（零歧义）
                info = dex_lookup(AUTHORITY[raw])
                mode = "authority"
            if not info:                               # 2) 标记价交叉验证自动去歧义
                q = SEARCH_NAME.get(raw, raw.lower())
                mark = fetch_mark_price(sym)
                info = dex_lookup_by_mark(q, mark)
                # 有标记价且成功比对 → mark；否则退化为「流动性最大」= 更存疑
                mode = "mark" if (mark and info and info.get("price_gap") is not None) else "fallback"
            if info:
                rows.append({"symbol": sym, "base": raw, "kind": "dex",
                             "chain": info["chain"], "contract_address": info["address"],
                             "source": f"dexscreener:{mode}",
                             "price_gap": info.get("price_gap"),
                             "_liq": info["liquidity"], "_vol": info["volume"],
                             "_mode": mode})
            else:
                rows.append({"symbol": sym, "base": raw, "kind": "unknown",
                             "chain": None, "contract_address": None, "source": None,
                             "price_gap": None})
    return rows


def cex_volume(symbol: str) -> float:
    """Binance 24h quoteVolume（CEX 流动性代理）。"""
    try:
        r = requests.get(f"{BASE}/api/v3/ticker/24hr", params={"symbol": symbol},
                         timeout=15, proxies=_proxies())
        return float(r.json().get("quoteVolume", 0) or 0)
    except Exception:
        return 0.0


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    db = MonitorDB(ncfg.DB_PATH)
    pool = (mcfg.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print(f"[liquidity] pool {len(pool)} 币，加载 Binance 现货映射…")
    coins = load_binance_coins()
    print(f"[liquidity] Binance 现货 {len(coins)} 币")

    rows = build_map(pool, coins)
    n_cex = sum(1 for r in rows if r["kind"] == "cex")
    n_dex = sum(1 for r in rows if r["kind"] == "dex")
    n_unk = sum(1 for r in rows if r["kind"] == "unknown")
    print(f"[liquidity] 映射：CEX {n_cex} | DEX {n_dex} | 未知 {n_unk}")

    for r in rows:
        db.upsert_token_map(r["symbol"], r["base"], r["kind"],
                            r.get("chain"), r.get("contract_address"), r.get("source"),
                            r.get("price_gap"))

    print("[liquidity] 采集流动性快照…")
    for r in rows:
        if r["kind"] == "cex":
            vol = cex_volume(r["symbol"])
            db.insert_liquidity(r["symbol"], "cex", None, vol, None, "binance")
        elif r["kind"] == "dex":
            db.insert_liquidity(r["symbol"], "dex", r.get("_liq"), r.get("_vol"), None, "dexscreener")
        time.sleep(0.15)

    # 打印结果
    print("\n=== 映射 + 流动性 ===")
    for r in rows:
        if r["kind"] == "cex":
            row = db.query("SELECT volume_24h FROM liquidity WHERE symbol=? ORDER BY ts DESC LIMIT 1",
                           (r["symbol"],))
            v = row[0][0] if row else 0
            print(f"  {r['symbol']:<14} CEX  24h成交额 ${(v or 0)/1e6:,.1f}M")
        elif r["kind"] == "dex":
            gap = r.get("price_gap")
            mode = r.get("_mode")
            if mode == "authority":
                status = "权威"
            elif gap is None:
                status = "存疑·无标记价"
            elif gap <= GAP_THRESHOLD:
                status = "高置信"
            else:
                status = "存疑"
            gap_s = f"  gap {gap*100:.1f}%" if gap is not None else ""
            addr = r.get("contract_address") or ""
            print(f"  {r['symbol']:<14} DEX  {str(r.get('chain') or '?'):<10} "
                  f"TVL ${(r.get('_liq') or 0):,.0f}  {addr[:10]}…  [{status}]{gap_s}")
        else:
            print(f"  {r['symbol']:<14} 未知（查不到）")
    db.close()


if __name__ == "__main__":
    main()
