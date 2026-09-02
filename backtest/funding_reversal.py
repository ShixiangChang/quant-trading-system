# -*- coding: utf-8 -*-
"""资金费率截面反向回测：费率极正→空、极负→多，捕捉「对手方被迫成交」的均值回归。

与 pin_strategy 同方法论（上升沿 + 去重 + 每事件固定仓位 + 参数扫描），
但信号源换成 funding（仓位拥挤度指标），验证第二条正交的赔率不对称腿。

机制：
- 费率极正 = 多头极度拥挤（多头付空头钱）→ 部分多头被强平/被迫撤 → 卖压 → 回调 → 反向空
- 费率极负 = 空头极度拥挤 → 空头被挤 → 反弹 → 反向多

信号：每个 funding 结算时刻（8h），对全池做截面 z 分数；|z| > 阈值 → 反向开仓。
  z 截面化（相对同期 peers），自然对冲市场整体 funding 漂移。
  上升沿触发（z 从阈值内→外的首个时刻）+ 同币 hold 期内去重，避免重叠样本陷阱。

数据：funding_hist（8h，~164 币 2 年）+ klines（1h 收盘）。
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd

DB = "data/monitor.db"
MIN_FUND_PTS = 500       # 币至少有 500 个 funding 点才进池（~5 个月）
MIN_POOL = 10            # 每个截面至少 10 币才算 z
Z_TH = 2.0               # 截面 z 阈值
HOLD_H = 24              # 持有小时数
COST_SIDE = 0.0006
OUT = "data/model_out/funding_reversal.json"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB)
    fund = pd.read_sql_query(
        "SELECT symbol, funding_time, funding FROM funding_hist", conn)
    price = pd.read_sql_query(
        "SELECT symbol, open_time, close FROM klines", conn)
    conn.close()
    # 池：funding 点数足够 + 有 1h K线
    fcnt = fund.groupby("symbol").size()
    pcount = price.groupby("symbol").size()
    pool = sorted(set(fcnt[fcnt >= MIN_FUND_PTS].index) & set(pcount.index))
    fund = fund[fund.symbol.isin(pool)]
    price = price[price.symbol.isin(pool)]
    return fund, price, pool


def main() -> None:
    global Z_TH, HOLD_H, COST_SIDE, OUT
    ap = argparse.ArgumentParser(description="资金费率截面反向回测")
    ap.add_argument("--z-th", type=float, default=Z_TH, help="截面 z 阈值")
    ap.add_argument("--hold", type=int, default=HOLD_H, help="持有小时数")
    ap.add_argument("--cost", type=float, default=COST_SIDE, help="单边成本")
    ap.add_argument("--dir", choices=["contrarian", "momentum"], default="contrarian",
                    help="方向：contrarian=极正空/极负多（反向）；momentum=极正多/极负空（顺势/携带）")
    ap.add_argument("--out", default=OUT, help="落盘路径")
    a = ap.parse_args()
    Z_TH = a.z_th
    HOLD_H = a.hold
    COST_SIDE = a.cost
    OUT = a.out
    DIR = 1.0 if a.dir == "contrarian" else -1.0

    fund, price, pool = load()

    # funding 截面：pivot funding_time × symbol
    fmat = fund.pivot_table(index="funding_time", columns="symbol",
                            values="funding", aggfunc="last").sort_index()
    # 截面 z（沿行，即每个时刻对全池标准化）
    fz = fmat.sub(fmat.mean(axis=1), axis=0).div(fmat.std(axis=1), axis=0)

    # 1h 价格：pivot open_time × symbol → close
    pmat = price.pivot_table(index="open_time", columns="symbol",
                             values="close", aggfunc="last").sort_index()
    p_ts = pmat.index.values.astype(np.int64)
    P = pmat.values.astype(float)

    f_ts = fz.index.values.astype(np.int64)
    Z = fz.values.astype(float)
    syms = list(fz.columns)
    T, N = Z.shape

    # 事件收集：上升沿 + 去重
    long_mask = Z < -Z_TH
    short_mask = Z > Z_TH
    last = np.full(N, -10**9, dtype=np.int64)      # 每币上次入场时间戳
    events = []                                     # (entry_ts, j, dir, ret)

    def px_at(t: int) -> np.ndarray:
        """每个币在时间 t 的 1h 收盘（t 之后第一根 1h bar 的 close，越界 NaN）。"""
        idx = np.searchsorted(p_ts, t, side="left")
        idx = np.clip(idx, 0, len(p_ts) - 1)
        return P[idx]

    for i in range(T):
        t = f_ts[i]
        # 上升沿：本次触发（long/short）且上次触发在 hold 期外
        trig = long_mask[i] | short_mask[i]
        if not trig.any():
            continue
        cols = np.where(trig)[0]
        for j in cols:
            if t - last[j] < HOLD_H * 3600:
                continue
            d = DIR * (-1.0 if short_mask[i, j] else 1.0)  # contrarian: 极正空/极负多；momentum 反向
            entry = px_at(t)[j]
            exit_ = px_at(t + HOLD_H * 3600)[j]
            if np.isnan(entry) or np.isnan(exit_) or entry <= 0 or exit_ <= 0:
                continue
            last[j] = t
            ret = d * (exit_ / entry - 1.0) - 2 * COST_SIDE
            events.append((t, j, d, ret))

    if not events:
        print("无事件（阈值太严或数据不足）")
        return

    rets = np.array([e[3] for e in events])
    n = len(rets)
    up = rets[rets > 0]
    dn = rets[rets < 0]
    p_up = len(up) / n
    U = up.mean() * p_up
    D = -dn.mean() * (1 - p_up)

    dir_label = "反向(极正空/极负多)" if DIR == 1.0 else "顺势(极正多/极负空)"
    print(f"=== 资金费率截面 {dir_label}（z>{Z_TH}/z<-{Z_TH} 触发 / 持有 {HOLD_H}h / "
          f"成本 {COST_SIDE:.4f} / 池 {len(pool)} 币） ===")
    print(f"独立事件 {n:,}，均值 {rets.mean()*100:+.3f}% 中位 {np.median(rets)*100:+.3f}% "
          f"命中率 {p_up:.1%} 赔率比 {U/D:.2f}")

    print("--- 事件级分年 ---")
    yearly = {}
    for y in [2024, 2025, 2026]:
        sub = np.array([e[3] for e in events
                        if pd.Timestamp(e[0], unit="s").year == y])
        if len(sub):
            up2 = sub[sub > 0]; dn2 = sub[sub < 0]
            U2 = up2.mean() * (len(up2)/len(sub)); D2 = -dn2.mean() * (len(dn2)/len(sub))
            print(f"  {y}: n={len(sub):>4} 均值{sub.mean()*100:+.2f}% "
                  f"命中{(sub>0).mean():.0%} 赔率比{U2/D2:.2f}")
            yearly[y] = {"n": int(len(sub)), "mean": float(sub.mean()),
                         "hit": float((sub > 0).mean())}

    # 组合级：每事件固定仓位 5%，按日聚合
    POS_SIZE = 0.05
    by_day: dict[int, list[float]] = {}
    for ts0, j, d, ret in events:
        by_day.setdefault(ts0 // 86400, []).append(ret)
    all_days = sorted(set(int(ts) // 86400 for ts in p_ts))
    dret = np.array([POS_SIZE * np.sum(by_day[d]) if d in by_day else 0.0
                     for d in all_days])
    eq = np.cumprod(1 + dret)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    ann = eq[-1] ** (365 / len(dret)) - 1
    sharpe = dret.mean() / (dret.std() + 1e-12) * np.sqrt(365)
    active = sum(1 for d in all_days if d in by_day)
    print(f"\n--- 组合级（每事件 {POS_SIZE:.0%} 仓位，{len(dret)} 天，{active} 天有持仓，"
          f"空仓 {(1-active/len(dret)):.0%}） ---")
    print(f"总收益 {eq[-1]-1:+.1%}  年化 {ann:+.1%}  Sharpe {sharpe:.2f}  "
          f"最大回撤 {dd.min()*100:.1f}%")

    # 正交性：与插针、慢动量 v2 的相关
    corr_pin = corr_v2 = None
    try:
        pin = json.load(open("data/model_out/pin_strategy_12h.json", encoding="utf-8"))
        pin_ret = {int(d): r for d, r in pin["equity"]}
        common = sorted(set(by_day) & set(pin_ret))
        p = np.array([POS_SIZE * np.sum(by_day[d]) for d in common])
        v = np.array([pin_ret[d] for d in common])
        corr_pin = float(np.corrcoef(p, v)[0, 1]) if len(common) > 5 else None
    except Exception:
        pass
    try:
        v2 = json.load(open("data/model_out/replay_all.json", encoding="utf-8"))["legs"]["v2"]["equity"]
        v2_map = {int(x[0]) // 86400: float(x[1]) for x in v2}
        v2_days = sorted(v2_map)
        v2_ret = {v2_days[i+1]: v2_map[v2_days[i+1]] / v2_map[v2_days[i]] - 1
                  for i in range(len(v2_days) - 1)}
        common = sorted(set(by_day) & set(v2_ret))
        p = np.array([POS_SIZE * np.sum(by_day[d]) for d in common])
        v = np.array([v2_ret[d] for d in common])
        corr_v2 = float(np.corrcoef(p, v)[0, 1]) if len(common) > 5 else None
    except Exception:
        pass
    print(f"与插针相关 {corr_pin}，与慢动量 v2 相关 {corr_v2}")

    payload = {
        "params": {"z_th": Z_TH, "hold_h": HOLD_H, "cost_side": COST_SIDE,
                   "pool": len(pool), "pos_size": POS_SIZE},
        "events": int(n), "mean_ret": float(rets.mean()),
        "median_ret": float(np.median(rets)), "hit_rate": float(p_up),
        "odds": float(U / D), "total_ret": float(eq[-1] - 1),
        "annual": float(ann), "sharpe": float(sharpe), "max_dd": float(dd.min()),
        "corr_vs_pin": corr_pin, "corr_vs_v2": corr_v2,
        "yearly": yearly,
        "equity": [[int(d), round(float(e), 4)] for d, e in zip(all_days, eq)],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已落盘 {OUT}")


if __name__ == "__main__":
    main()
