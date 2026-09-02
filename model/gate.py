# -*- coding: utf-8 -*-
"""验证闸门：DSR/PSR/MinTRL + 逐日 IC 序列（ICIR），回答「截面 z 信号是真 edge 还是选择偏差」。

蓝图第 1 步，也是唯一决定后续做不做的闸门。
站在两个巨人肩上：
- mlfinlab / de Prado《AFML》：PSR（含偏度/峰度修正的 Sharpe 显著性）、
  DSR（扣掉「N 次试验取最佳」的选择偏差）、MinTRL（结论所需最少样本）。
- qlib：逐日截面 IC 序列 + ICIR，不是全样本一个数。

口径：24h 截面 z 标签（fwd_ret_24h_cs）、z=1.0、净口径（扣换手+资金费，train.evaluate 已算）。

用法（在项目根目录下）:
    python -m model.gate                 # 57 池（probe_pool.txt）对照 18 蓝筹
    python -m model.gate --pool 18       # 只跑 18 蓝筹
    python -m model.gate --trials 40     # 无 sweep_results 记录时手动指定试验数 N
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from . import config, features, train


# ---------------- 统计闸门（mlfinlab 口径） ----------------

def _sr(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=0)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0,
                               steps_per_year: float = 1.0) -> float:
    """PSR = Φ[(SR̂−SR*)·√(T−1) / √(1−γ₃SR̂+(γ₄−1)/4·SR̂²)]，Lo(2002) 非正态修正。

    口径：SR̂/SR* 都用「年化 Sharpe」（与 sweep_results.csv 的 sharpe 列同尺度）。
    steps_per_year 把每步收益的 Sharpe 年化（96h → ×√91.25），否则会和年化基准比错尺度。
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < 3:
        return float("nan")
    sr = _sr(r) * np.sqrt(steps_per_year)   # 年化，对齐基准尺度
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # raw kurtosis，正态=3
    var_sr = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if var_sr <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(var_sr)))


def deflated_sharpe_ratio(returns, sr_trials, steps_per_year: float = 1.0) -> float:
    """DSR：SR* = std(trials)·[(1−γ_E)Φ⁻¹(1−1/N) + γ_E·Φ⁻¹(1−1/(Ne))]，γ_E=0.5772。"""
    r = np.asarray(returns, dtype=float)
    trials = np.asarray(sr_trials, dtype=float)
    N = len(trials)
    if N == 0 or len(r) < 3:
        return float("nan")
    gamma_e = 0.5772156649015329
    sr_star = trials.std(ddof=1) * (
        (1 - gamma_e) * norm.ppf(1 - 1.0 / N) + gamma_e * norm.ppf(1 - 1.0 / (N * np.e))
    )
    return probabilistic_sharpe_ratio(r, sr_star, steps_per_year)


def min_track_record_length(returns, sr_benchmark: float = 0.0, alpha: float = 0.05,
                            steps_per_year: float = 1.0) -> float:
    """MinTRL = 1 + [1−γ₃SR̂+(γ₄−1)/4·SR̂²]·[Φ⁻¹(1−α)/(SR̂−SR*)]²。结论所需最少独立样本（步）。"""
    r = np.asarray(returns, dtype=float)
    sr = _sr(r) * np.sqrt(steps_per_year)   # 年化，对齐基准尺度
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    num = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if sr <= sr_benchmark:
        return float("inf")
    return float(1 + num * (norm.ppf(1 - alpha) / (sr - sr_benchmark)) ** 2)


def _load_trials(fallback_n: int) -> list[float]:
    """历史所有试过的配置的 Sharpe（=选择偏差的 N）。优先 sweep_results.csv，没有则用占位并标注。"""
    path = config.OUTPUT_DIR / "sweep_results.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if "sharpe" in df.columns and len(df) > 0:
                return df["sharpe"].astype(float).tolist()
        except Exception:
            pass
    # 无历史记录：用 N 个近零占位（等于「假设各试验独立、真值≈0」的最保守基线），打印里标注。
    return [0.0] * fallback_n


# ---------------- 单池验证 ----------------

def _run_one(panel: pd.DataFrame, h: int, z: float, trials: list[float], tag: str) -> dict | None:
    label = f"fwd_ret_{h}h_cs"
    ret = f"fwd_ret_{h}h"
    feats = train.feature_cols(panel)
    res = train.evaluate(panel, feats, h, label, z, ret_col=ret)
    if res is None:
        print(f"[gate] {tag}: 无有效折")
        return None
    m = res["metrics"]
    port_ret = res["port"]["port_ret"].to_numpy()
    spy = 365 * 24 / h   # 年化因子：PSR/DSR 用年化 Sharpe，和 sweep_results 的 sharpe 同尺度
    di = res.get("daily_ic")
    if di is not None and len(di) > 1:
        icir = float(di.mean() / (di.std(ddof=1) + 1e-12))
        ic_pos = float((di > 0).mean())
    else:
        icir = float("nan")
        ic_pos = float("nan")
    return {
        "tag": tag,
        "ic": res["ic"],
        "icir": icir,
        "ic_pos": ic_pos,
        "ic_n": 0 if di is None else len(di),
        "sharpe": m["sharpe"],
        "total_ret": m["total_ret"],
        "max_dd": m["max_dd"],
        "n_steps": m["n"],
        "n_folds": res["n_folds"],
        "long_n": res["long_n"],
        "short_n": res["short_n"],
        "psr": probabilistic_sharpe_ratio(port_ret, 0.0, spy),
        "dsr": deflated_sharpe_ratio(port_ret, trials, spy),
        "mtrl": min_track_record_length(port_ret, 0.0, 0.05, spy),
    }


def _print_row(r: dict) -> None:
    print(f"{r['tag']:<8s} IC={r['ic']:+.4f} ICIR={r['icir']:+.2f} IC正日{r['ic_pos']:.0%} "
          f"净Sharpe={r['sharpe']:+.2f} 收益={r['total_ret']:+.1%} 回撤={r['max_dd']:.1%} "
          f"步数={r['n_steps']} 折={r['n_folds']}")
    print(f"           PSR={r['psr']:.2f} | DSR={r['dsr']:.2f} | MinTRL={r['mtrl']:.0f}步 "
          f"(多{r['long_n']} 空{r['short_n']})")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="验证闸门：DSR/PSR + OOS")
    p.add_argument("--pool", default="57", help="57 = probe_pool.txt；18 = 只跑蓝筹")
    p.add_argument("--trials", type=int, default=40, help="无 sweep 记录时的试验数 N")
    p.add_argument("--horizon", type=int, default=24, help="持仓期(h)，默认 24；96h 是持仓期地形图的最优点")
    a = p.parse_args()

    trials = _load_trials(a.trials)
    h = a.horizon
    print(f"[gate] DSR 试验数 N={len(trials)}"
          + ("（来自 sweep_results.csv）" if (config.OUTPUT_DIR / "sweep_results.csv").exists()
             else "（无历史记录，占位假设）"))

    results = []
    if a.pool != "18":
        pool_file = config.OUTPUT_DIR / "probe_pool.txt"
        pool = pool_file.read_text(encoding="utf-8").split()
        print(f"[gate] 构建 57 池面板（{len(pool)} 币）…")
        panel = features.build_panel(progress=False, symbols=pool, use_micro=False)
        print(f"[gate] 面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 币")
        r = _run_one(panel, h, 1.0, trials, f"57池{h}h")
        if r:
            results.append(r)

    if a.pool != "57":
        print("[gate] 构建 18 蓝筹面板…")
        panel18 = features.build_panel(progress=False, use_micro=False)
        r = _run_one(panel18, h, 1.0, trials, f"18蓝筹{h}h")
        if r:
            results.append(r)

    if not results:
        print("[gate] 无可验证结果")
        return

    print(f"\n=== 验证闸门结果（{h}h 截面 z，z=1.0，净口径）===")
    for r in results:
        _print_row(r)

    print("\n=== 结论 ===")
    for r in results:
        dsr = r["dsr"]
        if np.isnan(dsr):
            verdict = "数据不足，无法判定"
        elif dsr >= 0.95:
            verdict = "✅ 通过：DSR≥0.95，信号高于 N 次试验的期望最大，非纯选择偏差"
        else:
            verdict = "❗ 未通过：DSR<0.95，与选择偏差尚无法区分 → 只上纸面前向验证，暂不上实盘"
        print(f"  {r['tag']}: {verdict}（MinTRL={r['mtrl']:.0f}步，当前样本 {r['n_steps']} 步）")


if __name__ == "__main__":
    main()
