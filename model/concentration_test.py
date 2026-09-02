# -*- coding: utf-8 -*-
"""集中度因子粗筛：用免费代理指标验证「集中度」有没有 alpha。

- OI 集中度：某币美元 OI / 全市场 OI（截面占比，反映资金是否集中在少数大币）
- 大户持仓比 top_lsr：model/monitor 已有的 topLongShortPositionRatio（聪明钱多空方向）

数据：monitor.db 的 micro_1h（30 天）+ klines（算未来收益）。
输出：两个特征对 fwd_24h/fwd_96h 的截面 rank-IC（均值 + ICIR）。
诚实说明：micro 只有 30 天历史，IC 统计意义偏弱，这里是「粗筛方向」，不是最终结论。

用法: python -m model.concentration_test
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from monitor import config as ncfg


def load() -> pd.DataFrame:
    db = sqlite3.connect(ncfg.DB_PATH)
    micro = pd.read_sql(
        "SELECT symbol, ts, kind, value FROM micro_1h WHERE kind IN ('oi','top_lsr')", db)
    oi = micro[micro.kind == 'oi'][['symbol', 'ts', 'value']].rename(columns={'value': 'oi'})
    top = micro[micro.kind == 'top_lsr'][['symbol', 'ts', 'value']].rename(columns={'value': 'top_lsr'})
    df = oi.merge(top, on=['symbol', 'ts'], how='inner')

    kl = pd.read_sql("SELECT symbol, open_time, close FROM klines ORDER BY symbol, open_time", db)
    db.close()

    # 对齐：micro ts 取整到小时，与 klines open_time 对齐
    df['ts'] = (df['ts'] // 3600) * 3600
    df = df.merge(kl, left_on=['symbol', 'ts'], right_on=['symbol', 'open_time'], how='inner')

    # OI 集中度（美元口径 = 币本位 OI × 价格，再算截面占比）
    df['oi_notional'] = df['oi'] * df['close']
    df['oi_conc'] = df['oi_notional'] / df.groupby('ts')['oi_notional'].transform('sum')

    # 未来收益
    for h in (24, 96):
        df[f'fwd_{h}'] = df.groupby('symbol')['close'].shift(-h) / df['close'] - 1
    return df


def cross_sectional_ic(df: pd.DataFrame, feat: str, label: str) -> tuple[float, float]:
    ics = []
    for _, g in df.groupby('ts'):
        if len(g) >= 8:
            ic = g[feat].corr(g[label], method='spearman')
            if pd.notna(ic):
                ics.append(ic)
    ics = np.array(ics)
    return float(ics.mean()), float(ics.std())


def main() -> None:
    df = load()
    print(f"[conc] panel {len(df):,} 行 | {df['symbol'].nunique()} 币 | "
          f"{df['ts'].nunique()} 个时间截面")
    print("\n=== 截面 rank-IC（特征 vs 未来收益，30 天粗筛） ===")
    print(f"{'特征':<12} {'标签':<8} {'IC':>8} {'std':>8} {'ICIR':>7}")
    for feat in ('oi_conc', 'top_lsr'):
        for label in ('fwd_24', 'fwd_96'):
            m, s = cross_sectional_ic(df, feat, label)
            print(f"{feat:<12} {label:<8} {m:>+8.4f} {s:>8.4f} {m/s:>+7.2f}")


if __name__ == '__main__':
    main()
