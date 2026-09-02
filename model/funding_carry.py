# -*- coding: utf-8 -*-
"""资金费率拥挤 carry：透明规则测试（无训练、无过拟合，天然样本外）。

假设（加密量化文献性 edge，站在巨人肩上）：极端正资金费 = 拥挤杠杆多头，会被清算下跌 → 做空；
极端负资金费 = 拥挤空头，会回补反弹 → 做多。24h 持有。

诚实口径：简单收益 expm1(log_ret)，扣双边成本，加资金费 carry（做空收、做多付）。
与 ML 截面 z 信号无关，是另一条独立信号线——诊断已证明 z 信号的多头是毒药，换这条试试。

用法: python -m model.funding_carry
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, features


def _sr(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=0)
    return float(r.mean() / sd) if sd > 0 else 0.0


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print(f"[fc] pool {len(pool)} 币，构建面板…")
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)

    h = 24
    fwd_col = f"fwd_ret_{h}h"
    panel = panel.dropna(subset=[fwd_col, "funding"]).copy()
    # 简单收益（钱）
    panel["_r"] = np.expm1(panel[fwd_col])

    # 非重叠 24h 步（对齐 _fold 的 step 采样，避免重叠仓）
    times = pd.Series(panel["open_time"].unique()).sort_values().reset_index(drop=True)
    step_times = set(times.iloc[::h])
    p = panel[panel["open_time"].isin(step_times)]

    # 每步截面按资金费排名：top-k 做空、bottom-k 做多
    K = 5
    rows = []
    for ts, g in p.groupby("open_time"):
        g = g.sort_values("funding", ascending=False)
        short = g.head(K)          # 资金费最高 → 做空
        long = g.tail(K)           # 资金费最低 → 做多
        if len(short) < K or len(long) < K:
            continue
        price = (-short["_r"].mean() + long["_r"].mean()) / 2.0     # 各占 50% 权重
        carry = (-short["funding"].mean() * (h / 8.0) + long["funding"].mean() * (h / 8.0)) / 2.0  # 空收多付
        # 简化：carry 直接是每步 net 的增量（做空高资金费收钱、做多低资金费付钱，取平均）
        net = price + carry - config.COST_SIDE * 2
        rows.append({"ts": ts, "price": price, "carry": carry, "net": net})

    df = pd.DataFrame(rows)
    if df.empty:
        print("[fc] 无有效步")
        return
    steps_per_year = 365 * 24 / h
    print("\n=== 资金费率拥挤 carry（Top5 做空 / Bottom5 做多，24h，K=5） ===")
    print(f"步数 {len(df)} | 年化步 {steps_per_year:.0f}")
    print(f"价格(毛) Sharpe = {_sr(df['price'].to_numpy()) * np.sqrt(steps_per_year):+.2f}  | 平均 {df['price'].mean():+.3%}/步")
    print(f"资金费 carry   = 平均 {df['carry'].mean():+.3%}/步（这是真金白银的 carry，非预测）")
    print(f"净 Sharpe      = {_sr(df['net'].to_numpy()) * np.sqrt(steps_per_year):+.2f}  | 总收益 {float(np.prod(1 + df['net']) - 1):+.1%}")
    print(f"净回撤         = {float((np.cumprod(1 + df['net']) / np.maximum.accumulate(np.cumprod(1 + df['net'])) - 1).min()):.1%}")

    # 阈值敏感性：K=3/5/10
    print("\n=== K 敏感性（净 Sharpe） ===")
    for k in (3, 5, 10):
        rr = []
        for ts, g in p.groupby("open_time"):
            g = g.sort_values("funding", ascending=False)
            if len(g) < 2 * k:
                continue
            sh, lo = g.head(k), g.tail(k)
            price = (-sh["_r"].mean() + lo["_r"].mean()) / 2.0
            carry = (-sh["funding"].mean() + lo["funding"].mean()) * (h / 8.0) / 2.0
            rr.append(price + carry - config.COST_SIDE * 2)
        rr = np.asarray(rr)
        print(f"  K={k:>2}: 净 Sharpe {_sr(rr) * np.sqrt(steps_per_year):+.2f}  | 平均 {rr.mean():+.3%}/步")


if __name__ == "__main__":
    main()
