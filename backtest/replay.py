# -*- coding: utf-8 -*-
"""历史回放引擎：把库里躺着的 2 年 1h K线一次性重放，产出海量样本 + 完整净值曲线。

为什么需要它：
- 病根：137 万条 1h K线（166 币 × 733 天）躺在库里，但实时预测只用了最后 4 天 33 个截面，
  净值曲线只有 3 个点（4 条水平线）。等于把 2 年数据浪费了 98%。
- 量化公司的「量」从来不是「日历时间攒样本」，是「历史数据回放 × 横截面 × 并行」：
  Two Sigma 380PB/每天 10 万次模拟，Renaissance 每天数万笔/分钟级持仓。
- 正解：733 天 × 每天一个截面 × 未来 96h 结算 = 一次跑完 2 年，
  立刻得到 ~12 万因子样本 + 733 点完整净值曲线，几分钟跑完。

无前视（walk-forward）纪律：
- 因子（动量/ATR/趋势 z/波动率 z）严格用 t 之前的数据算；
- 结算严格用 t 之后的数据（未来 high/low/close）；
- 趋势 z 与波动率 z 的标定标准差用 expanding std（只用 t 之前），不偷看全历史。

用法：
    python backtest/replay.py            # 多腿对比：beta vs 慢动量（含波动率门控开关）
    python backtest/replay.py --save     # 结果落库 data/model_out/replay_*.json 供看板展示
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path("data/monitor.db")
OUT = Path("data/model_out")

# —— 参数与 decision/momentum_leg.py 对齐（2026-09-01 回看 7天→30天、top10→20）——
MOM_LOOKBACK_H = 30 * 24   # 回看 30 天（walk-forward 样本外验证优于 7 天）
MOM_SKIP_H = 24            # skip 最近 1 天
MOM_HOLD_H = 96            # 持有 96h（与系统持有期一致）
MOM_TOP_N = 20             # 做多最强势 N 个（分散）
MOM_POS_MAX = 0.05         # 每币 5%，20 币等权满仓 = 100%
MOM_TREND_OFFSET = 1.0     # 趋势门控偏移（z>1.0 才有效开仓，回撤≤40%）
ATR_MULT = 3.0             # 止损距离 3×ATR
TREND_MA_HOURS = 720       # 趋势均线 30 日


def load_klines() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT symbol, open_time, high, low, close FROM klines ORDER BY open_time, symbol",
        conn,
    )
    conn.close()
    return df


def prep_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """向量化预计算因子 + 未来结算价 + 趋势/波动率 z。返回 (df, trend_z, vol_z)。"""
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    # —— 每币 logret ——
    df["logret"] = df.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))

    # —— 动量：多 lookback（7天=v1 旧参数，30天=v2 新参数），严格 t 之前 ——
    df["mom_168"] = df.groupby("symbol")["logret"].transform(
        lambda s: s.rolling(168).sum().shift(MOM_SKIP_H))
    df["mom_720"] = df.groupby("symbol")["logret"].transform(
        lambda s: s.rolling(720).sum().shift(MOM_SKIP_H))

    # —— ATR14 归一化 ——
    tr = df["high"] - df["low"]
    df["atr"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14).mean())
    df["atr_norm"] = df["atr"] / df["close"]

    # —— 未来 hold 期间 high/low/close（严格 t 之后）——
    df = df.iloc[::-1]
    df["f_low"] = df.groupby("symbol")["low"].transform(
        lambda s: s.shift(1).rolling(MOM_HOLD_H, min_periods=1).min())
    df["f_high"] = df.groupby("symbol")["high"].transform(
        lambda s: s.shift(1).rolling(MOM_HOLD_H, min_periods=1).max())
    df["f_close"] = df.groupby("symbol")["close"].transform(
        lambda s: s.shift(MOM_HOLD_H))
    df = df.iloc[::-1]

    # —— 全池等权几何指数 ——
    idx_ret = df.groupby("open_time")["logret"].mean()
    idx = np.exp(idx_ret.cumsum())

    # —— 趋势 z（expanding std，只用 t 之前标定）——
    ma = idx.rolling(TREND_MA_HOURS).mean()
    dev = idx / ma - 1.0
    trend_z = dev / dev.expanding(min_periods=TREND_MA_HOURS).std()

    # —— 波动率 z（指数 24h 收益滚动 std，相对自身历史标准化，expanding）——
    vol = idx.pct_change().rolling(24).std()
    vm = vol.expanding(min_periods=30).mean()
    vs = vol.expanding(min_periods=30).std()
    vol_z = (vol - vm) / vs

    df["trend_z"] = df["open_time"].map(trend_z.to_dict())
    df["vol_z"] = df["open_time"].map(vol_z.to_dict())

    # —— 指数未来 96h 收益（beta 腿用）——
    idx_fwd = idx.shift(-MOM_HOLD_H) / idx - 1.0
    df["idx_fwd"] = df["open_time"].map(idx_fwd.to_dict())

    return df, trend_z, vol_z


def _vol_scale(vz: float) -> float:
    """波动率门控（连续）：vol_z≤0 全仓，vol_z≥2 缩到 0.5。"""
    z = float(vz) if pd.notna(vz) else 0.0
    if z <= 0:
        return 1.0
    return max(0.5, 1.0 - 0.25 * z)


def replay_momentum(df: pd.DataFrame, vol_gate: bool = False,
                     lookback_h: int = 720, top_n: int = 20,
                     offset: float = 1.0, pos_max: float = 0.05,
                     start_ts: int | None = None, end_ts: int | None = None,
                     score_col: str | None = None) -> dict:
    """慢动量腿全历史回放。参数化支持 v1(7天)/v2(30天) 对比 + 指定时间段独立轨道。

    score_col 为 None 时按 lookback 选 mom_168/mom_720；传入则用任意评分列排序 top_n
    （供深度动量网络 DMN 分走同一条回放管线，与 v2 公平对打）。
    """
    if score_col is None:
        score_col = "mom_168" if lookback_h == 168 else "mom_720"
    d = df.dropna(subset=[score_col, "atr_norm", "trend_z"]).copy()
    cross = d[d["open_time"] % 86400 == 0].sort_values("open_time")
    times = sorted(cross["open_time"].unique())
    if start_ts is not None:
        times = [t for t in times if t >= start_ts]
    if end_ts is not None:
        times = [t for t in times if t <= end_ts]
    cross = cross[cross["open_time"].isin(times)]

    equity, trades = [], []
    nav = 1.0

    for t in times:
        snap = cross[cross["open_time"] == t]
        z = float(snap["trend_z"].iloc[0])
        w = max(0.0, np.tanh(z - offset))
        if w <= 1e-6:
            equity.append([t, nav])
            continue
        vz = float(snap["vol_z"].iloc[0]) if vol_gate else 0.0
        vscale = _vol_scale(vz) if vol_gate else 1.0
        top = snap.nlargest(top_n, score_col)
        leg_ret, n = 0.0, 0
        for _, r in top.iterrows():
            entry = float(r["close"])
            atr = max(float(r["atr_norm"]) or 0.03, 0.005)
            stop = entry * (1.0 - ATR_MULT * atr)
            if pd.isna(r["f_close"]):
                continue
            exit_px = stop if (pd.notna(r["f_low"]) and r["f_low"] <= stop) else float(r["f_close"])
            hit = exit_px == stop
            one = exit_px / entry - 1.0
            trades.append({"ts": t, "symbol": r["symbol"], "ret": one,
                           "hit_stop": hit, "trend_z": z, "vol_z": vz})
            leg_ret += one * pos_max * vscale
            n += 1
        if n:
            nav *= (1.0 + leg_ret)
        equity.append([t, nav])

    rets = [tr["ret"] for tr in trades]
    hits = sum(1 for r in rets if r > 0)
    stops = sum(1 for tr in trades if tr["hit_stop"])
    eq = pd.DataFrame(equity, columns=["ts", "nav"])
    return {
        "equity": equity,
        "stats": {
            "n_trades": len(trades),
            "n_cross": len(times),
            "hit_rate": hits / len(trades) if trades else 0,
            "avg_ret": float(np.mean(rets)) if rets else 0,
            "total_ret": float(eq["nav"].iloc[-1] - 1) if len(eq) else 0,
            "stop_rate": stops / len(trades) if trades else 0,
            "max_drawdown": float(_max_dd(eq["nav"])) if len(eq) else 0,
        },
    }


def replay_beta(df: pd.DataFrame) -> dict:
    """beta 腿全历史回放：指数 > 30日均线（trend_z>0）→ 满仓持全池等权指数，否则空仓。"""
    d = df.dropna(subset=["trend_z", "idx_fwd"]).copy()
    cross = d[d["open_time"] % 86400 == 0].sort_values("open_time")
    times = sorted(cross["open_time"].unique())

    equity, nav = [], 1.0
    for t in times:
        snap = cross[cross["open_time"] == t]
        z = float(snap["trend_z"].iloc[0])
        if z > 0:
            r = float(snap["idx_fwd"].iloc[0])   # 全池等权指数未来 96h 收益
            nav *= (1.0 + r)
        equity.append([t, nav])

    eq = pd.DataFrame(equity, columns=["ts", "nav"])
    return {
        "equity": equity,
        "stats": {
            "n_cross": len(times),
            "total_ret": float(eq["nav"].iloc[-1] - 1) if len(eq) else 0,
            "max_drawdown": float(_max_dd(eq["nav"])) if len(eq) else 0,
        },
    }


def _max_dd(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float(((nav - peak) / peak).min())


def _fmt(stats: dict, has_hr: bool = True) -> str:
    parts = [f"累计 {stats['total_ret']*100:+.1f}%", f"回撤 {stats['max_drawdown']*100:.1f}%"]
    if has_hr:
        parts += [f"命中 {stats['hit_rate']*100:.1f}%", f"止损 {stats['stop_rate']*100:.1f}%",
                  f"{stats['n_trades']}笔/{stats['n_cross']}截面"]
    return "  |  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--start", help="独立轨道起始日期 YYYY-MM-DD")
    ap.add_argument("--end", help="独立轨道结束日期 YYYY-MM-DD")
    ap.add_argument("--track", help="独立轨道名称（保存到 data/model_out/tracks/<name>.json）")
    a = ap.parse_args()

    print("加载 klines ...")
    df = load_klines()
    print(f"  {len(df):,} 条 1h K线, {df['symbol'].nunique()} 币")

    print("预计算因子（无前视）...")
    df, trend_z, vol_z = prep_features(df)

    # —— 独立轨道模式：指定时间段，跑一条独立轨道，与现有状态无关 ——
    if a.track:
        if not a.start or not a.end:
            print("错误：--track 需要同时指定 --start 和 --end")
            return
        start_ts = int(datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        res = replay_momentum(df, lookback_h=720, top_n=20, offset=1.0, pos_max=0.05,
                              start_ts=start_ts, end_ts=end_ts)
        eq = pd.DataFrame(res["equity"], columns=["ts", "nav"])
        # 分年/分月
        eq["ym"] = pd.to_datetime(eq["ts"], unit="s").dt.to_period("M").astype(str)
        monthly = {ym: float(g["nav"].iloc[-1] / g["nav"].iloc[0] - 1) for ym, g in eq.groupby("ym")}
        track = {
            "name": a.track,
            "start": a.start, "end": a.end,
            "params": {"lookback_h": 720, "top_n": 20, "offset": 1.0, "pos_max": 0.05, "atr_mult": ATR_MULT},
            "equity": res["equity"],
            "monthly": monthly,
            "stats": res["stats"],
        }
        tracks_dir = OUT / "tracks"
        tracks_dir.mkdir(exist_ok=True)
        (tracks_dir / f"{a.track}.json").write_text(json.dumps(track, default=str), encoding="utf-8")
        st = res["stats"]
        print(f"\n独立轨道 [{a.track}] {a.start} ~ {a.end} 跑完:")
        print(f"  累计 {st['total_ret']*100:+.1f}% | 回撤 {st['max_drawdown']*100:.1f}% | "
              f"命中 {st['hit_rate']*100:.1f}% | {st['n_trades']}笔/{st['n_cross']}截面")
        print(f"  已落库 {tracks_dir}/{a.track}.json")
        return

    print("\n" + "=" * 70)
    print("多腿 2 年全历史回放对比（同一市场环境，净值差异 = 纯策略差异）")
    print("=" * 70)

    beta = replay_beta(df)
    # v1 旧参数：7天回看 + top10 + 无偏移 + 每币10%
    v1 = replay_momentum(df, lookback_h=168, top_n=10, offset=0.0, pos_max=0.10)
    # v2 新参数：30天回看 + top20 + 偏移1.0 + 每币5%
    v2 = replay_momentum(df, lookback_h=720, top_n=20, offset=1.0, pos_max=0.05)

    print(f"\n  {'β 纯趋势跟随(基准)':<20} {_fmt(beta['stats'], has_hr=False)}")
    print(f"  {'慢动量v1·7天(已证伪)':<20} {_fmt(v1['stats'])}")
    print(f"  {'慢动量v2·30天(最优)':<20} {_fmt(v2['stats'])}")

    # 分年
    print("\n  分年收益:")
    for name, res in [("beta", beta), ("v1·7天", v1), ("v2·30天", v2)]:
        eq = pd.DataFrame(res["equity"], columns=["ts", "nav"])
        eq["y"] = pd.to_datetime(eq["ts"], unit="s").dt.year
        yrs = [f"{y}:{(g['nav'].iloc[-1]/g['nav'].iloc[0]-1)*100:+.0f}%" for y, g in eq.groupby("y")]
        print(f"    {name:<10} " + "  ".join(yrs))

    print("\n" + "=" * 70)

    if a.save:
        OUT.mkdir(exist_ok=True)
        payload = {
            "legs": {
                "beta": beta,
                "v1": v1,
                "v2": v2,
            }
        }
        (OUT / "replay_all.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
        (OUT / "replay_momentum.json").write_text(json.dumps(v2, default=str), encoding="utf-8")
        print(f"\n已落盘 data/model_out/replay_all.json + replay_momentum.json")


if __name__ == "__main__":
    main()
