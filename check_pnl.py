# -*- coding: utf-8 -*-
"""一条命令看当前所有持仓盈亏，实时 mark price 计价。

用法（在项目根目录下）：
    python check_pnl.py
    python check_pnl.py --full    # 附带第一批次逐币明细
"""
from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
import urllib.request


def _fetch_prices() -> dict:
    """拉全市场 mark price（走 Clash 代理）。失败返回空。"""
    proxy = urllib.request.ProxyHandler({"http": "http://127.0.0.1:7890",
                                         "https": "http://127.0.0.1:7890"})
    opener = urllib.request.build_opener(proxy)
    try:
        data = json.loads(opener.open("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=20).read())
        return {d["symbol"]: float(d["markPrice"]) for d in data}
    except Exception as exc:
        print(f"[check_pnl] 拉取实时价失败: {exc}")
        return {}


def _ts2s(t):
    return datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M") if t else ""


def live_pnl(price: dict) -> None:
    conn = sqlite3.connect("data/monitor.db")
    cur = conn.cursor()
    cur.execute("SELECT id, symbol, side, entry_price, leverage, stop_price, size "
                "FROM live_trades WHERE status='open' ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("无持仓")
        return
    print("=" * 64)
    print("当前持仓（未平仓，实时 mark price 计价）")
    print("=" * 64)
    total = 0.0
    for tid, sym, side, entry, lev, stop, size in rows:
        now = price.get(sym)
        if now is None:
            print(f"  {sym}: 无实时价")
            continue
        chg = now / entry - 1
        pnl_lev = side * chg * lev        # 保证金口径（杠杆后）
        pnl_cap = pnl_lev * size          # 占总资金口径
        total += pnl_cap
        hit = (side > 0 and now <= stop) or (side < 0 and now >= stop)
        tag = "  ← 已触及止损!" if hit else ""
        print(f"  {sym:10} {'多' if side > 0 else '空'} 开{entry:<11} 现{now:<11.6g} "
              f"涨跌{chg:+.2%} | 保证金{pnl_lev:+.1%} | 占总资金{pnl_cap:+.1%}{tag}")
    print(f"  —— 合计占总资金：{total:+.1%}")


def paper_pnl(price: dict, full: bool) -> None:
    try:
        st = json.load(open("data/paper_state.json"))
    except Exception:
        print("纸面：无状态文件")
        return
    batches = st.get("batches", [])
    if not batches:
        print("纸面：无批次")
        return
    print()
    print("=" * 64)
    print("纸面批次（美元中性多空对冲，实时价计 gross）")
    print("=" * 64)
    for i, b in enumerate(batches):
        pos = b.get("positions", {})
        ep = b.get("entry_prices", {})
        gross = sum(w * (price[s] / ep[s] - 1) for s, w in pos.items()
                    if s in ep and s in price)
        print(f"  批次{i}（{_ts2s(b['ts'])} 开仓，{len(pos)} 币）gross = {gross:+.2%}")

    if full and batches:
        print()
        print("第一批次逐币明细：")
        b0 = batches[0]
        pos = b0["positions"]
        ep = b0["entry_prices"]
        tot = 0.0
        for s, w in sorted(pos.items(), key=lambda x: -x[1]):
            now = price.get(s)
            if now is None:
                continue
            ret = now / ep[s] - 1
            tot += w * ret
            print(f"  {s:10} {'多' if w > 0 else '空'} 开{ep[s]:<12} 现{now:<12.6g} "
                  f"涨跌{ret:+.2%} 贡献{w * ret:+.2%}")
        print(f"  第一批次满仓净值 = {1 + tot:.4f}")


def four_plans_pnl(price: dict) -> None:
    """四套方案纸面仓（paper_four.py 开的仓）的当前盈亏。"""
    try:
        st = json.load(open("data/model_out/four_plans_state.json"))
    except Exception:
        return
    plans = st.get("plans", {})
    if not plans:
        return
    print()
    print("=" * 64)
    print("四套方案纸面仓（paper_four.py 开仓，实时价计）")
    print("=" * 64)
    for plan, p in plans.items():
        hold = p.get("holdings", {})
        pnl = sum(v["side"] * (price.get(s, 0.0) / v["price"] - 1.0) * v["pos"]
                  for s, v in hold.items() if s in price and v["price"] > 0)
        longs = sum(1 for v in hold.values() if v["side"] > 0)
        shorts = sum(1 for v in hold.values() if v["side"] < 0)
        print(f"  {plan:10} {len(hold)} 仓（多{longs}/空{shorts}）pnl = {pnl:+.2%}")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description="查当前持仓盈亏")
    p.add_argument("--full", action="store_true", help="附带第一批次逐币明细")
    a = p.parse_args()
    price = _fetch_prices()
    if not price:
        print("无法获取实时价，请检查网络/代理")
        return
    live_pnl(price)
    paper_pnl(price, a.full)
    four_plans_pnl(price)


if __name__ == "__main__":
    main()
