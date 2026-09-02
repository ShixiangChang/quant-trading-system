# -*- coding: utf-8 -*-
"""训练与回测：walk-forward 滚动验证 + LightGBM + 绩效指标。

两层接口：
- walk_forward(panel)：打印版（model.main 用）
- evaluate(panel, feats, horizon, label_col, z)：参数化版（sweep 用），返回指标 + IC + 组合序列
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config

EXCLUDE = {"symbol", "open_time", "fwd_ret"}

# 原始 K 线水平列（价格/成交量/成交笔数）：无截面信号，只当「币种身份代理」——
# 价格 10 万 vs 0.009 直接告诉树模型「这是 BTC 还是 SKR」，模型背币名、样本内虚高、实盘会失效。
# 且价格水平非平稳（BTC 一路涨），分裂阈值会过期。必须剔除，只留无量纲相对特征。
RAW_LEVEL_COLS = {
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote",
}

# 时间哑变量（hour/dow 的 sin/cos）：诊断（model/time_diag.py）确认——删掉后 IC 反而
# +0.005、回撤 -45%→-40%，属纯过拟合（模型在背「几点涨周几涨」而非横截面强弱），剔除。
TIME_COLS = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}


def feature_cols(panel: pd.DataFrame) -> list[str]:
    # fwd_ret_* 是多周期标签列（fwd_ret_1h ... fwd_ret_48h 及其市场中性变体），必须排除，否则泄漏
    return [c for c in panel.columns
            if c not in EXCLUDE and c not in RAW_LEVEL_COLS and c not in TIME_COLS
            and not c.startswith("fwd_ret_")]


def _metrics(port_ret: np.ndarray, steps_per_year: float) -> dict:
    n = len(port_ret)
    if n == 0:
        return {"sharpe": 0.0, "total_ret": 0.0, "max_dd": 0.0,
                "win_rate": 0.0, "pf": 0.0, "n": 0}
    eq = np.cumprod(1.0 + port_ret)
    total = float(eq[-1] - 1.0)
    sd = float(port_ret.std(ddof=0))
    sharpe = float(port_ret.mean() / sd * np.sqrt(steps_per_year)) if sd > 0 else 0.0
    max_dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    wins = port_ret[port_ret > 0]
    losses = port_ret[port_ret < 0]
    win_rate = float(len(wins) / max(1, int((port_ret != 0).sum())))
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else float("inf")
    return {"sharpe": sharpe, "total_ret": total, "max_dd": max_dd,
            "win_rate": win_rate, "pf": pf, "n": n}


def _make_folds(panel: pd.DataFrame) -> list:
    """生成 walk-forward 折的 (开始, 训练止, 测试止) 时间戳。"""
    t0 = panel["open_time"].min()
    t1 = panel["open_time"].max()
    folds, t = [], t0
    while t + pd.Timedelta(days=config.TRAIN_DAYS + config.TEST_DAYS) <= t1:
        folds.append((t, t + pd.Timedelta(days=config.TRAIN_DAYS),
                      t + pd.Timedelta(days=config.TRAIN_DAYS + config.TEST_DAYS)))
        t += pd.Timedelta(days=config.STEP_DAYS)
    return folds


def _fold(panel, tr_mask, te_mask, feats, fold_id, label_col, ret_col, horizon, z):
    tr = panel[tr_mask]
    te = panel[te_mask]
    if len(tr) < config.MIN_TRAIN_ROWS:
        return None

    dtr = lgb.Dataset(tr[feats].to_numpy(dtype=float), label=tr[label_col].to_numpy(dtype=float))
    Xte = te[feats].to_numpy(dtype=float)

    # 多 seed 集成：big-model 单次训练方差大（net Sharpe 0.71~2.55 摇摆），
    # 用不同 seed 各训一棵、预测取平均，把单模型方差压下去（qlib 大模型的标配做法）。
    preds = []
    scores = np.zeros(len(feats))
    for k in range(config.ENSEMBLE_SEEDS):
        params = dict(config.LGBM_PARAMS)
        params["seed"] = config.LGBM_PARAMS["seed"] + k
        booster = lgb.train(params, dtr, num_boost_round=config.NUM_BOOST_ROUND)
        preds.append(booster.predict(Xte))
        scores += booster.feature_importance("gain")
    importance = dict(zip(feats, scores))

    te = te.copy()
    te["pred"] = np.mean(preds, axis=0)

    # 每 horizon 根 K 线取一行（步进 = 持仓期，避免仓位重叠）
    times = pd.Series(te["open_time"].unique()).sort_values().reset_index(drop=True)
    step_times = set(times.iloc[::horizon])
    sub = te[te["open_time"].isin(step_times)].sort_values(["symbol", "open_time"])

    # 截面标准化预测值：|z| 超过阈值才下单，其余空仓
    sub["pred_z"] = sub.groupby("open_time")["pred"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12)
    )
    sub["sig"] = 0
    sub.loc[sub["pred_z"] >= z, "sig"] = 1
    sub.loc[sub["pred_z"] <= -z, "sig"] = -1
    sub["sig_prev"] = sub.groupby("symbol")["sig"].shift(1).fillna(0)
    sub["turnover"] = (sub["sig"] - sub["sig_prev"]).abs()
    # 毛收益必须用「简单收益率」记账（钱）。ret_col 是 fwd_ret_*h 对数收益，必须 expm1 转简单收益
    # 再复利：cumprod(1+log_ret) 在 log_ret<-1 时净值变负、Sharpe/回撤全崩（-99.99% 回撤的根因）。
    # 也不能用训练标签（截面 z 分数无量纲、均值≈0），拿它复利同样滚出天文数字。
    sub["gross"] = sub["sig"] * np.expm1(sub[ret_col])
    # 资金费（近似）：进场费率 × 持仓期结算次数（8h 一结）；多头付正费、空头收。
    # 雏形阶段用 as-of 进场费率近似；精确逐笔累计留给纸面引擎。
    if "funding" in sub.columns:
        sub["funding_cost"] = -(sub["sig"] * sub["funding"].fillna(0.0)) * (horizon / 8.0)
    else:
        sub["funding_cost"] = 0.0
    sub["net"] = sub["gross"] - config.COST_SIDE * sub["turnover"] + sub["funding_cost"]

    return {
        "fold": fold_id,
        "sub": sub,                 # 保留 pred / 实际 / sig / net，供 IC 与拼组合
        "importance": importance,
        "n_train": len(tr),
        "n_test": len(sub),
    }


def evaluate(panel: pd.DataFrame, feats: list[str], horizon: int,
             label_col: str, z: float, ret_col: str | None = None) -> dict | None:
    """参数化 walk-forward。返回 {metrics, port, ic, ic_per_fold, n_folds, long_n, short_n} 或 None。

    label_col = 训练标签（模型学什么，可为截面 z / 市场中性 / 原始收益）
    ret_col   = 记账收益（钱怎么算，必须是收益率，默认取 label_col 以兼容旧调用）
    """
    if ret_col is None:
        ret_col = label_col
    folds = _make_folds(panel)
    results, all_port, all_gross, ics, daily_ics = [], [], [], [], []
    for i, (s, e_tr, e_te) in enumerate(folds):
        purge = pd.Timedelta(hours=horizon)  # 训练段末尾裁掉 horizon 根 bar：其标签窗口伸进测试期，属前视泄漏
        tr_mask = (panel["open_time"] >= s) & (panel["open_time"] < e_tr - purge)
        te_mask = (panel["open_time"] >= e_tr) & (panel["open_time"] < e_te)
        res = _fold(panel, tr_mask, te_mask, feats, i, label_col, ret_col, horizon, z)
        if res is None:
            continue
        results.append(res)
        port = (res["sub"].groupby("open_time")["net"].mean()
                .rename("port_ret").reset_index().sort_values("open_time"))
        all_port.append(port)
        port_gross = (res["sub"].groupby("open_time")["gross"].mean()
                      .rename("port_ret").reset_index().sort_values("open_time"))
        all_gross.append(port_gross)
        ics.append(res["sub"]["pred"].corr(res["sub"][label_col], method="spearman"))
        # 逐日（每持仓步）截面 rank-IC：qlib 口径的 IC 序列，供 ICIR / 稳定性判断，非全样本一个数
        for _ts, _g in res["sub"].groupby("open_time"):
            if len(_g) >= 4:
                daily_ics.append((_ts, _g["pred"].corr(_g[label_col], method="spearman")))

    if not results:
        return None

    concat = pd.concat(all_port, ignore_index=True)
    concat_gross = pd.concat(all_gross, ignore_index=True)
    steps_per_year = 365 * 24 / horizon
    overall = _metrics(concat["port_ret"].to_numpy(), steps_per_year)           # 净口径：扣换手 + 资金费
    overall_gross = _metrics(concat_gross["port_ret"].to_numpy(), steps_per_year)  # 毛口径：未扣成本

    all_sub = pd.concat([r["sub"] for r in results], ignore_index=True)
    ic_pool = all_sub["pred"].corr(all_sub[label_col], method="spearman")

    return {
        "metrics": overall,
        "metrics_gross": overall_gross,
        "port": concat,
        "ic": float(ic_pool) if pd.notna(ic_pool) else 0.0,
        "ic_per_fold": [float(x) for x in ics if pd.notna(x)],
        "n_folds": len(results),
        "long_n": int((all_sub["sig"] == 1).sum()),
        "short_n": int((all_sub["sig"] == -1).sum()),
        "daily_ic": pd.Series([v for _, v in daily_ics], index=[t for t, _ in daily_ics]),
    }


def walk_forward(panel: pd.DataFrame) -> None:
    """打印版（model.main 用），内部走 evaluate + 逐折复算打印。"""
    feats = feature_cols(panel)
    folds = _make_folds(panel)
    print(f"\n[model] 共 {len(folds)} 折 | 特征 {len(feats)} 个 | 预测 {config.HORIZON}h 收益")
    print(f"[model] 成本 {config.COST_SIDE:.2%}/边 | 截面 z 阈值 ±{config.SIGNAL_Z}（|z|<阈值空仓）")

    results, all_port, all_imp = [], [], {}
    for i, (s, e_tr, e_te) in enumerate(folds):
        purge = pd.Timedelta(hours=config.HORIZON)
        tr_mask = (panel["open_time"] >= s) & (panel["open_time"] < e_tr - purge)
        te_mask = (panel["open_time"] >= e_tr) & (panel["open_time"] < e_te)
        res = _fold(panel, tr_mask, te_mask, feats, i, "fwd_ret", "fwd_ret", config.HORIZON, config.SIGNAL_Z)
        if res is None:
            print(f"[model] 折 {i} 训练样本不足，跳过")
            continue
        results.append(res)
        all_port.append(res["sub"].groupby("open_time")["net"].mean().rename("port_ret").reset_index())
        for k, v in res["importance"].items():
            all_imp[k] = all_imp.get(k, 0.0) + v

    if not results:
        print("[model] 没有可回测的折")
        return

    # 逐折打印
    print("\n=== 各折回测 ===")
    print(f"{'折':>2} {'训练':>7} {'测试':>7} {'多':>5} {'空':>5} {'Sharp':>7} {'收益':>7} {'回撤':>7} {'胜率':>6} {'盈亏':>6} {'IC':>6}")
    for r in results:
        m = _metrics(r["sub"].groupby("open_time")["net"].mean().to_numpy(), 365 * 24 / config.HORIZON)
        ic = r["sub"]["pred"].corr(r["sub"]["fwd_ret"], method="spearman")
        print(f"{r['fold']:>2} {r['n_train']:>7} {r['n_test']:>7} "
              f"{(r['sub']['sig'] == 1).sum():>5} {(r['sub']['sig'] == -1).sum():>5} "
              f"{m['sharpe']:>7.2f} {m['total_ret']:>7.1%} {m['max_dd']:>7.1%} {m['win_rate']:>6.1%} {m['pf']:>6.2f} "
              f"{ic:>6.3f}")

    # 合计
    concat = pd.concat(all_port, ignore_index=True).sort_values("open_time")
    overall = _metrics(concat["port_ret"].to_numpy(), 365 * 24 / config.HORIZON)
    benchmark = _benchmark(panel, folds)

    print("\n=== 合计（所有折拼接，连续复利） ===")
    print(f"年化Sharpe {overall['sharpe']:.2f} | 总收益 {overall['total_ret']:.1%} | "
          f"最大回撤 {overall['max_dd']:.1%} | 步胜率 {overall['win_rate']:.1%} | 盈亏比 {overall['pf']:.2f} | 步数 {overall['n']}")
    print(f"基准(等权买入持有) Sharpe {benchmark['sharpe']:.2f} | 总收益 {benchmark['total_ret']:.1%} | "
          f"最大回撤 {benchmark['max_dd']:.1%}")

    # 特征重要性
    imp = pd.Series(all_imp).sort_values(ascending=False)
    print("\n=== 特征重要性 Top 15（gain 合计） ===")
    for feat, val in imp.head(15).items():
        print(f"  {feat:<20s} {val:,.0f}")

    # 落盘
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    concat.to_csv(config.OUTPUT_DIR / "equity_curve.csv", index=False)
    imp.to_csv(config.OUTPUT_DIR / "feature_importance.csv")
    print(f"\n[model] 输出已保存到 {config.OUTPUT_DIR}")


def _benchmark(panel: pd.DataFrame, folds: list) -> dict:
    """等权买入持有基准：测试区间内每步的平均未来收益。"""
    mask = pd.Series(False, index=panel.index)
    for s, e_tr, e_te in folds:
        mask |= (panel["open_time"] >= e_tr) & (panel["open_time"] < e_te)
    bench = panel[mask]
    if bench.empty:
        return _metrics(np.array([]), 365 * 24 / config.HORIZON)
    times = pd.Series(bench["open_time"].unique()).sort_values().reset_index(drop=True)
    step_times = set(times.iloc[::config.HORIZON])
    bench = bench[bench["open_time"].isin(step_times)]
    port = bench.groupby("open_time")["fwd_ret"].mean().sort_index()
    return _metrics(port.to_numpy(), 365 * 24 / config.HORIZON)