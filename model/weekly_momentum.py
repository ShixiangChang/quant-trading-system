# -*- coding: utf-8 -*-
"""跨截面周级动量（Liu & Tsyvinski 2021）：过去 1/2/4 周收益 → 未来 1 周收益。透明规则，无训练无过拟合。

诊断证明 24h 是「均值回归」区（追涨做空弱正 +0.2%/天、抄底做多强负 -0.4%/天）；文献里加密动量在周级成立。
这个脚本测：把持仓/回看拉到周级，信号会不会从「回归」翻成「动量」（多赢家、空输家，且赢家延续）。

诚实口径：简单收益，扣双边成本（周级摊薄后成本几乎可忽略）。

用法: python -m model.weekly_momentum
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


def _rank_ic(rank: pd.Series, fwd: pd.Series) -> float:
    if len(rank) < 4:
        return float("nan")
    v = rank.corr(fwd, method="spearman")
    return float(v) if pd.notna(v) else float("nan")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    pool = (config.OUTPUT_DIR / "probe_pool.txt").read_text(encoding="utf-8").split()
    print(f"[wm] pool {len(pool)} 币，构建面板…")
    panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    # 过去收益（简单收益），跨币可比
    for n in (168, 336, 672):
        panel[f"past_{n}h"] = panel.groupby("symbol")["close"].pct_change(n)
    panel["fwd_168h_simple"] = np.expm1(panel["fwd_ret_168h"])

    hold = 168
    times = pd.Series(panel["open_time"].unique()).sort_values().reset_index(drop=True)
    step_times = set(times.iloc[::hold])
    p = panel[panel["open_time"].isin(step_times)].dropna(subset=["fwd_168h_simple"])

    K = 5
    steps_per_year = 365 * 24 / hold
    print(f"\n=== 周级截面动量（过去 N 周 → 未来 1 周，K={K}，净口径） ===")
    print(f"{'回看':>6} {'rank-IC':>9} {'毛Sharpe':>9} {'净Sharpe':>9} {'总收益':>8} {'回撤':>8} {'步数':>5}")

    for n in (168, 336, 672):
        past = f"past_{n}h"
        ics, nets, grosses = [], [], []
        for ts, g in p.groupby("open_time"):
            g = g.dropna(subset=[past, "fwd_168h_simple"])
            if len(g) < 2 * K:
                continue
            g = g.sort_values(past, ascending=False)
            top = g.head(K)      # 过去涨最多 → 做多（动量假设）
            bot = g.tail(K)      # 过去跌最多 → 做空
            if len(top) < K or len(bot) < K:
                continue
            rank = g[past].rank(pct=True)
            fwd = g["fwd_168h_simple"]
            ics.append(_rank_ic(rank, fwd))
            gross = top["fwd_168h_simple"].mean() - bot["fwd_168h_simple"].mean()  # 美元中性，gross 2x
            net = gross - 2 * config.COST_SIDE  # 双边成本 0.12%，每周一次
            grosses.append(gross)
            nets.append(net)

        ics = np.asarray(ics); nets = np.asarray(nets); grosses = np.asarray(grosses)
        eq = np.cumprod(1 + nets)
        dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        print(f"{n / 24:>6.0f}周 {np.nanmean(ics):>+9.4f} {_sr(grosses) * np.sqrt(steps_per_year):>+9.2f} "
              f"{_sr(nets) * np.sqrt(steps_per_year):>+9.2f} {float(eq[-1] - 1):>+8.1%} {dd:>8.1%} {len(nets):>5}")


if __name__ == "__main__":
    main()
