# -*- coding: utf-8 -*-
"""流动性映射摸底：对 pool 里的币，用 DexScreener 定位现货 token 的链 + 合约地址 + 流动性。

输出：覆盖率表（定位 / 低流动性 / 查不到）+ 流动性分档。
这是「拿数据策略」第一步：验证免费源能覆盖多少币、哪些币是高风险薄流动性。

用法: python -m model.liquidity_map
"""
from __future__ import annotations

import sys
import time

import requests

from . import config

# 合约 symbol 与现货搜索名不一致的特殊映射
SPECIAL = {
    "龙虾": "lobster",
    "1000PEPE": "pepe",
}

LIQ_TIER = [
    (100_000_000, "高流动性"),
    (10_000_000, "中流动性"),
    (1_000_000, "低流动性"),
    (100_000, "极低流动性"),
    (0, "微量/死币"),
]


def to_query(sym: str) -> str:
    base = sym[:-4] if sym.endswith("USDT") else sym
    return SPECIAL.get(base, base.lower())


def tier(liq: float) -> str:
    for th, name in LIQ_TIER:
        if liq >= th:
            return name
    return "微量/死币"


def fetch_one(sym: str) -> dict:
    q = to_query(sym)
    try:
        r = requests.get("https://api.dexscreener.com/latest/dex/search",
                         params={"q": q}, timeout=20)
        pairs = r.json().get("pairs", [])
    except Exception as e:
        return {"sym": sym, "q": q, "status": "error", "err": str(e)}
    if not pairs:
        return {"sym": sym, "q": q, "status": "查不到"}
    best = max(pairs, key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0)
    liq = best.get("liquidity", {}).get("usd", 0) or 0
    return {
        "sym": sym, "q": q, "status": "定位" if liq >= 100_000 else "低流动性",
        "chain": best.get("chainId"),
        "address": (best.get("baseToken") or {}).get("address"),
        "liquidity": liq,
        "volume24h": (best.get("volume") or {}).get("h24", 0),
        "fdv": best.get("fdv", 0),
        "tier": tier(liq),
    }


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    rows = []
    for i, sym in enumerate(pool, 1):
        r = fetch_one(sym)
        rows.append(r)
        liq = r.get("liquidity", 0) or 0
        print(f"[{i}/{len(pool)}] {r['sym']:<14} -> {r['status']:<6} "
              f"{r.get('tier', '')}  ${liq:,.0f}")
        time.sleep(0.25)

    ok = sum(1 for r in rows if r["status"] == "定位")
    low = sum(1 for r in rows if r["status"] == "低流动性")
    miss = sum(1 for r in rows if r["status"] in ("查不到", "error"))
    print(f"\n=== 覆盖率 === 定位 {ok} | 低流动性 {low} | 查不到/错误 {miss} | 共 {len(rows)}")

    print("\n=== 高风险/查不到清单 ===")
    for r in rows:
        if r["status"] != "定位":
            print(f"  {r['sym']:<14} {r['status']:<8} {r.get('tier', '')} liq=${r.get('liquidity', 0) or 0:,.0f}")

    print("\n=== 完整表（按流动性降序） ===")
    for r in sorted(rows, key=lambda x: x.get("liquidity", 0) or 0, reverse=True):
        print(f"  {r['sym']:<14} {str(r.get('chain') or '?'):<10} ${(r.get('liquidity', 0) or 0):>14,.0f}  "
              f"{r.get('tier', '')}")


if __name__ == "__main__":
    main()
