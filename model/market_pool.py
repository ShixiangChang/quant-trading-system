# -*- coding: utf-8 -*-
"""生成全市场流动性池清单：拉 Binance 全市场 24h ticker，按成交额降序取 Top N。
用成交额（quoteVolume）当流动性代理，天然过滤死币/零成交币，且覆盖大部分暴涨暴跌 mover。

用法（在项目根目录下）:
    python -m model.market_pool 200            # 生成 top 200 池 -> market_pool_200.txt
    python -m model.market_pool                 # 默认 200
"""
from __future__ import annotations

import sys

from . import config, data


def build_pool(top_n: int = 200) -> str:
    tickers = data._get_json("/fapi/v1/ticker/24hr")
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):        # 只要 USDT 本位永续
            continue
        qv = t.get("quoteVolume", 0.0)
        if qv is None:
            continue
        rows.append((sym, float(qv)))
    rows.sort(key=lambda r: r[1], reverse=True)
    pool = [s for s, _ in rows[:top_n]]
    path = config.OUTPUT_DIR / f"market_pool_{top_n}.txt"
    path.write_text("\n".join(pool), encoding="utf-8")
    print(f"[market_pool] 全市场 {len(rows)} 个 USDT 永续，成交额 Top {len(pool)} 写入 {path.name}")
    for i, (s, qv) in enumerate(rows[:top_n], 1):
        print(f"  {i:>3} {s:>18} {qv/1e8:>8.1f} 亿")
    return pool


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    build_pool(n)