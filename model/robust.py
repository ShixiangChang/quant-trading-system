# -*- coding: utf-8 -*-
"""上线前最后一关（v2）：48h 信号到底是什么 —— 真实可解释的，还是 ML 凹出来的。

v1 发现朴素"动量"基线 rank-IC 为负（反转，非动量），推翻了我"截面动量"的说法。
v1 还有个采样 bug：朴素多空收益没做非重叠 48h 采样，-100% 那个数不可信（重叠自相关虚标步数）。
这一版修正：非重叠 48h 采样 + 逐个特征 rank-IC 定位真实机制 + 动量/反转两个朴素规则对照。

结论纪律：解释得清、简单规则能复现 → 下纸面交易；解释不清 → 老实砍掉重来。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from model import config, features, train


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    panel = features.build_panel(progress=False)
    panel["fwd_ret_48h_mn"] = panel["fwd_ret_48h"] - panel.groupby("open_time")["fwd_ret_48h"].transform("mean")

    # 非重叠 48h 采样（与 ML 回测一致），避免重叠自相关
    times = pd.Series(panel["open_time"].unique()).sort_values().reset_index(drop=True)
    step_times = set(times.iloc[::48])
    sub = panel[panel["open_time"].isin(step_times)].copy()

    # ---- 1) 逐个特征 rank-IC（定位机制与符号） ----
    feats = ["ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_48h",
             "rsi14", "bb_pctb", "dist_high48", "dist_low48",
             "close_sma20", "close_sma50", "close_sma200", "vol_24h"]
    print("=== 特征 rank-IC（非重叠 48h，对 fwd_ret_48h_mn；符号=方向）===")
    for f in feats:
        if f not in sub.columns:
            continue
        tmp = sub.dropna(subset=[f, "fwd_ret_48h_mn"]).copy()
        if len(tmp) < 200:
            continue
        tmp["_r"] = tmp.groupby("open_time")[f].rank(pct=True)
        ic = tmp["_r"].corr(tmp["fwd_ret_48h_mn"], method="spearman")
        print(f"  {f:<14s} {ic:+.4f}")

    # ---- 2) 朴素规则：动量(多赢家) vs 反转(多输家)，非重叠 48h ----
    base = sub.dropna(subset=["ret_48h", "fwd_ret_48h"]).copy()
    base["mom_rank"] = base.groupby("open_time")["ret_48h"].rank(pct=True)
    sp = 365 * 24 / 48
    for name, long_top in [("动量(多赢家)", True), ("反转(多输家)", False)]:
        s = base.copy()
        s["sig"] = 0
        if long_top:
            s.loc[s["mom_rank"] >= 0.8, "sig"] = 1
            s.loc[s["mom_rank"] <= 0.2, "sig"] = -1
        else:
            s.loc[s["mom_rank"] <= 0.2, "sig"] = 1
            s.loc[s["mom_rank"] >= 0.8, "sig"] = -1
        ls = s[s["sig"] != 0]
        pnl = (ls["sig"] * ls["fwd_ret_48h"]).groupby(ls["open_time"]).mean()
        m = train._metrics(pnl.to_numpy(), sp)
        print(f"[naive] {name:<10s} Sharpe={m['sharpe']:+.2f} 总收益={m['total_ret']:+.1%} "
              f"回撤={m['max_dd']:.1%} 步数={m['n']}")

    # ---- 3) LightGBM 种子稳定性 ----
    feats_all = train.feature_cols(panel)
    for seed in (42, 7, 123):
        config.LGBM_PARAMS["seed"] = seed
        res = train.evaluate(panel, feats_all, 48, "fwd_ret_48h_mn", 1.0)
        if res:
            print(f"[lgb] seed={seed:<4} IC={res['ic']:+.4f} Sharpe={res['metrics']['sharpe']:+.2f}")
    config.LGBM_PARAMS["seed"] = 42

    # ---- 4) 上线前最后一道关卡：多重检验 / deflated Sharpe ----
    _deflated_gate(sub, sp)


# ---------------- 透明规则 + 多重检验关卡 ----------------
def _transparent_rule(sub: pd.DataFrame, sp: float):
    """透明规则：做多低波动+近48h高点、做空相反（前后20%），多空等额。

    返回 (metrics, rank_ic, R, S)。R/S 是 (step×symbol) 矩阵，供置换检验复用——
    观察策略与随机策略必须同结构、同目标，唯一的区别是「信号是否真的预测未来」。
    """
    sub = sub.copy()
    g = sub.groupby("open_time")
    sub["z_vol"] = g["vol_24h"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    sub["z_high"] = g["dist_high48"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    sub["score"] = -sub["z_vol"] + sub["z_high"]
    sub["rank"] = sub.groupby("open_time")["score"].rank(pct=True)
    sub["sig"] = 0
    sub.loc[sub["rank"] >= 0.8, "sig"] = 1
    sub.loc[sub["rank"] <= 0.2, "sig"] = -1

    sub["_r"] = sub.groupby("open_time")["score"].rank(pct=True)
    ic = sub["_r"].corr(sub["fwd_ret_48h_mn"], method="spearman")

    ls = sub[sub["sig"] != 0]
    pnl = (ls["sig"] * ls["fwd_ret_48h"]).groupby(ls["open_time"]).mean()
    m = train._metrics(pnl.to_numpy(), sp)

    R = sub.pivot(index="open_time", columns="symbol", values="fwd_ret_48h").fillna(0.0)
    S = sub.pivot(index="open_time", columns="symbol", values="sig").fillna(0.0)
    return m, float(ic), R, S


def _permute_sharpe(rng, Rmat: np.ndarray, Smat: np.ndarray,
                    n_active: np.ndarray, sp: float) -> float:
    """随机策略：每步把多空标签随机重排（保持每步多空数量不变），算其 Sharpe。

    这等价于「信号与未来收益毫无关联」的原假设下的一个策略样本。
    """
    order = np.argsort(rng.random(Smat.shape), axis=1)
    Sn = np.take_along_axis(Smat, order, axis=1)
    step = (Sn * Rmat).sum(axis=1) / n_active
    sd = step.std(ddof=0)
    return float(step.mean() / sd * np.sqrt(sp)) if sd > 0 else 0.0


def _deflated_gate(sub: pd.DataFrame, sp: float) -> None:
    """上线前最后一道统计关卡：这 +2.7 的 Sharpe，是本事还是 36 次试出来的运气？

    - 置换检验：模拟「36 次随机策略里取最佳」的 Sharpe 分布，看随机的最佳能否追上观察值。
    - Bonferroni：rank-IC 的 t 值做多重检验校正（36 次，已属保守——因子本身也是同一样本挖的）。
    """
    from math import erfc, sqrt as msqrt
    N_TRIALS = 36       # sweep 试过的组合数（6 horizon × 2 目标 × 3 阈值）
    B = 1000            # 置换组数（「取36个最佳」重复 B 次，估计其分布）

    m, ic, R, S = _transparent_rule(sub, sp)
    Rmat = R.to_numpy()
    Smat = S.to_numpy()
    n_active = (Smat != 0).sum(axis=1)
    obs_sharpe = m["sharpe"]

    rng = np.random.default_rng(12345)
    best = np.empty(B)
    for b in range(B):
        mx = -np.inf
        for _ in range(N_TRIALS):
            mx = max(mx, _permute_sharpe(rng, Rmat, Smat, n_active, sp))
        best[b] = mx

    p_perm = float((best >= obs_sharpe).mean())
    e_max = float(best.mean())

    n_steps = Rmat.shape[0]
    t = ic * msqrt(n_steps)
    p_two = float(erfc(abs(t) / msqrt(2.0)))
    p_bonf = min(1.0, p_two * N_TRIALS)

    print("\n=== 上线前最后一道关卡：多重检验 / deflated Sharpe ===")
    print(f"透明规则 rank-IC = {ic:+.4f} | t = {t:+.2f} | 双侧 p = {p_two:.3f} | Bonferroni({N_TRIALS}) p = {p_bonf:.3f}")
    print(f"观察组合 Sharpe = {obs_sharpe:+.2f} | 总收益 {m['total_ret']:+.1%} | 回撤 {m['max_dd']:.1%} | 步数 {m['n']}")
    print(f"{B} 组「{N_TRIALS} 个随机策略取最佳」的 Sharpe：中位 {np.median(best):+.2f} | P95 {np.percentile(best, 95):+.2f} | 最大 {best.max():+.2f}")
    print(f"P(随机最佳 ≥ {obs_sharpe:+.2f}) = {p_perm:.3f}  →  "
          f"{'❗ 与噪声无法区分' if p_perm > 0.05 else '✅ 明显高于噪声'}")
    print(f"经验 deflated Sharpe = {obs_sharpe - e_max:+.2f}（观察 − 随机最佳期望 {e_max:+.2f}）")
    if p_perm <= 0.05 and obs_sharpe - e_max > 0:
        print("结论：样本内站得住，但样本量薄、且因子来自同一样本，最终需经纸面/实盘验证。")
    else:
        print("结论：无法与噪声区分，建议换因子或调低预期。")
    print()


if __name__ == "__main__":
    main()