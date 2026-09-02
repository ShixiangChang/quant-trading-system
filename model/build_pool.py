# -*- coding: utf-8 -*-
"""扩池：拉 Binance 全量 USDT 永续，按流动性门槛 + 异动币补充，重写 probe_pool.txt。

截面 z 信号吃的是「截面宽度」（IR = IC × √Breadth），每多一个币 = 多一个独立下注 + 截面更稳。
但薄盘币价差宽/易插针/易下架，必须用流动性门槛挡在门外。异动币(暴涨暴跌)是事件驱动 track 的
原料（爆仓潮/OI 异动/资金费极值都发生在它们身上），单独补充，不计流动性下限。

用法：
    python -m model.build_pool           # 用默认门槛重写 probe_pool.txt
    python -m model.build_pool --vol 20e6 --mover 15 --max 250
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config
from .data import _get_json

VOLUME_FLOOR = 10_000_000   # 24h 名义成交额下限（美元）
MOVER_CHG = 15.0            # 24h 涨跌幅 |%| 达到即纳入（异动币，不看流动性）
MAX_POOL = 250              # 池子上限（面板构建成本可控）


def build_pool(vol_floor: float = VOLUME_FLOOR, mover_chg: float = MOVER_CHG,
               max_pool: int = MAX_POOL, top_n: int | None = None) -> pd.DataFrame:
    info = _get_json("/fapi/v1/exchangeInfo")
    syms = [s["symbol"] for s in info["symbols"]
            if s["symbol"].endswith("USDT")
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"]

    tick = _get_json("/fapi/v1/ticker/24hr")
    tset = set(syms)
    rows = []
    for t in tick:
        if t["symbol"] not in tset:
            continue
        try:
            qv = float(t["quoteVolume"])
        except (TypeError, ValueError):
            continue
        try:
            chg = float(t["priceChangePercent"])
        except (TypeError, ValueError):
            chg = 0.0
        rows.append({"symbol": t["symbol"], "quote_vol": qv, "chg": chg})

    df = pd.DataFrame(rows).sort_values("quote_vol", ascending=False).reset_index(drop=True)
    if top_n is not None:
        # 精选模式：只取成交额 Top N（流动性核心）。扩池实验已证明：截面 z 信号在
        # 流动性核心、稀释到长尾反而变差（Sharpe 1.69→0.95），所以长尾是噪声不是 Breadth。
        pool = df.head(top_n)
    else:
        liquid = df[df["quote_vol"] >= vol_floor]
        movers = df[df["chg"].abs() >= mover_chg]
        pool = pd.concat([liquid, movers]).drop_duplicates("symbol")
        pool = pool.sort_values("quote_vol", ascending=False).reset_index(drop=True)
        if len(pool) > max_pool:
            pool = pool.head(max_pool)
    return pool, df


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="扩池：全量 USDT 永续 → 流动性门槛 → probe_pool.txt")
    p.add_argument("--vol", type=float, default=VOLUME_FLOOR, help="24h 成交额下限(美元)")
    p.add_argument("--mover", type=float, default=MOVER_CHG, help="异动币涨跌幅门槛 %")
    p.add_argument("--max", type=int, default=MAX_POOL, help="池子上限")
    p.add_argument("--top", type=int, default=None, help="精选模式：只取成交额 Top N")
    a = p.parse_args()

    pool, df = build_pool(a.vol, a.mover, a.max, a.top)
    print(f"[pool] USDT 永续总数: {len(df)}")
    for thr in (100e6, 50e6, 20e6, 10e6, 5e6, 1e6):
        print(f"[pool]   24h vol >= ${thr/1e6:.0f}M : {(df['quote_vol'] >= thr).sum()} coins")
    print(f"[pool] movers |chg|>={a.mover}% : {int((df['chg'].abs() >= a.mover).sum())}")
    print(f"[pool] final pool: {len(pool)} coins "
          f"(vol_floor=${a.vol/1e6:.0f}M, min_vol=${pool['quote_vol'].min()/1e6:.1f}M)")

    out = config.OUTPUT_DIR / "probe_pool.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(pool["symbol"].tolist()), encoding="utf-8")
    print(f"[pool] wrote {out}")
    print("[pool] top10:", " ".join(pool["symbol"].head(10).tolist()))
    print("[pool] tail5 :", " ".join(pool["symbol"].tail(5).tolist()))


if __name__ == "__main__":
    main()
