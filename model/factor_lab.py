# -*- coding: utf-8 -*-
"""因子工厂（Factor Lab）：标准因子源 + 批量体检。

把「加一个因子」从「改 features.py 那个 200 行大函数」变成「写一个返回 (symbol, ts, value)
的函数 + 在 FACTOR_SOURCES 注册一行」。体检 = 对齐 klines 收盘价 → 算未来收益 → 截面 rank-IC
（均值 + ICIR），批量出报告。

因子源分两类（体检时各自用自己的覆盖范围，报告里标「覆盖天数」）：
- kline 因子：从 klines 表算，2 年历史（价量）
- micro 因子：从 micro_1h 表算，30 天历史（OI/多空比/主动买卖，含流动性代理）

用法: python -m model.factor_lab
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd

from monitor import config as ncfg

DB = ncfg.DB_PATH


# ------------------------------------------------ 数据读取
_KLINES: pd.DataFrame | None = None


def _read_klines() -> pd.DataFrame:
    """klines 表（symbol, open_time, close）读一次缓存，避免每个因子重读 137 万行。"""
    global _KLINES
    if _KLINES is None:
        db = sqlite3.connect(DB)
        _KLINES = pd.read_sql(
            "SELECT symbol, open_time, close FROM klines ORDER BY symbol, open_time", db)
        db.close()
    return _KLINES


def _kline_frame() -> pd.DataFrame:
    """因子函数用的独立副本（避免污染缓存），只取三列。"""
    return _read_klines()[["symbol", "open_time", "close"]].copy()


def _read_micro(kind: str) -> pd.DataFrame:
    db = sqlite3.connect(DB)
    df = pd.read_sql("SELECT symbol, ts, value FROM micro_1h WHERE kind=? ORDER BY symbol, ts",
                     db, params=(kind,))
    db.close()
    return df


# ------------------------------------------------ 因子源（每个返回 DataFrame[symbol, ts, value]）
def factor_ret_24h() -> pd.DataFrame:
    kl = _kline_frame()
    kl["value"] = kl.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(24)))
    return kl[["symbol", "open_time", "value"]].rename(columns={"open_time": "ts"}).dropna()


def factor_vol_24h() -> pd.DataFrame:
    kl = _kline_frame()
    kl["ret1"] = kl.groupby("symbol")["close"].transform(
        lambda s: np.log(s) - np.log(s.shift(1)))
    kl["value"] = kl.groupby("symbol")["ret1"].transform(lambda s: s.rolling(24).std())
    return kl[["symbol", "open_time", "value"]].rename(columns={"open_time": "ts"}).dropna()


def factor_rsi14() -> pd.DataFrame:
    kl = _kline_frame()

    def rsi(s: pd.Series) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0)
        l = -d.clip(upper=0)
        ag = g.ewm(alpha=1 / 14, min_periods=14).mean()
        al = l.ewm(alpha=1 / 14, min_periods=14).mean()
        return 100 - 100 / (1 + ag / al)

    kl["value"] = kl.groupby("symbol")["close"].transform(rsi)
    return kl[["symbol", "open_time", "value"]].rename(columns={"open_time": "ts"}).dropna()


def factor_oi_conc() -> pd.DataFrame:
    """OI 集中度：币美元 OI（币本位 OI × 价格）/ 全市场 OI（截面占比）。"""
    oi = _read_micro("oi")
    kl = _read_klines()
    kl = kl[["symbol", "open_time", "close"]].copy()
    kl["ts"] = (kl["open_time"] // 3600) * 3600
    df = oi.merge(kl[["symbol", "ts", "close"]], on=["symbol", "ts"], how="inner")
    df["notional"] = df["value"] * df["close"]
    df["value"] = df["notional"] / df.groupby("ts")["notional"].transform("sum")
    return df[["symbol", "ts", "value"]]


def factor_top_lsr() -> pd.DataFrame:
    return _read_micro("top_lsr")


def factor_lsr() -> pd.DataFrame:
    return _read_micro("lsr")


def factor_taker_bs() -> pd.DataFrame:
    return _read_micro("taker_bs")


# ------------------------------------------------ 注册表（加因子 = 加一行）
FACTOR_SOURCES = {
    "ret_24h": factor_ret_24h,
    "vol_24h": factor_vol_24h,
    "rsi14": factor_rsi14,
    "oi_conc": factor_oi_conc,
    "top_lsr": factor_top_lsr,
    "lsr": factor_lsr,
    "taker_bs": factor_taker_bs,
}

HORIZONS = (24, 96)
MIN_CROSS = 8   # 单个时间截面至少 8 币才算一个 IC 点


# ------------------------------------------------ 体检
def _cross_ic(df: pd.DataFrame, feat: str, label: str) -> np.ndarray:
    """向量化截面 rank-IC：返回每个时间截面的 spearman(feat, label)。

    rank 化后 spearman == pearson，用 groupby.transform(mean) + groupby.sum 一次算完，
    替代逐截面 Python 循环的 spearman（那是 2 年体检 3 分钟的元凶）。
    """
    tmp = pd.DataFrame({"ts": df["ts"], "f": df[feat], "l": df[label]}).dropna()
    size = tmp.groupby("ts")["f"].transform("count")
    tmp = tmp[size >= MIN_CROSS]
    if len(tmp) < MIN_CROSS:
        return np.array([])
    ts = tmp["ts"]
    fr = tmp.groupby("ts")["f"].rank()
    lr = tmp.groupby("ts")["l"].rank()
    fr_dm = fr - fr.groupby(ts).transform("mean")
    lr_dm = lr - lr.groupby(ts).transform("mean")
    num = (fr_dm * lr_dm).groupby(ts).sum()
    den = ((fr_dm ** 2).groupby(ts).sum() * (lr_dm ** 2).groupby(ts).sum()) ** 0.5
    ic = num / den.replace(0, np.nan)
    return ic.dropna().to_numpy()


def evaluate(factor_name: str, horizons: tuple[int, ...] = HORIZONS) -> dict:
    """体检单个因子：对齐 close → 未来收益 → 截面 rank-IC（均值 + ICIR）。"""
    factor = FACTOR_SOURCES[factor_name]()          # symbol, ts, value
    factor["ts"] = (factor["ts"] // 3600) * 3600
    kl = _read_klines()
    df = factor.merge(kl[["symbol", "open_time", "close"]],
                      left_on=["symbol", "ts"], right_on=["symbol", "open_time"], how="inner")
    for h in horizons:
        df[f"fwd_{h}"] = df.groupby("symbol")["close"].shift(-h) / df["close"] - 1

    out = {
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "days": round(df["ts"].nunique() / 24, 1),
    }
    for h in horizons:
        ics = _cross_ic(df, "value", f"fwd_{h}")
        if len(ics):
            out[f"ic_{h}"] = float(ics.mean())
            out[f"icir_{h}"] = float(ics.mean() / (ics.std() + 1e-12))
        else:
            out[f"ic_{h}"] = float("nan")
            out[f"icir_{h}"] = float("nan")
    return out


def run_all(horizons: tuple[int, ...] = HORIZONS) -> None:
    hdr = "  ".join(f"{h}h_IC  {h}h_ICIR" for h in horizons)
    print(f"{'因子':<12} {'覆盖':<10} {'行数':>9} {'币数':>6}  {hdr}")
    print("-" * 70)
    for name in FACTOR_SOURCES:
        try:
            r = evaluate(name, horizons)
            cells = "  ".join(f"{r[f'ic_{h}']:+.4f} {r[f'icir_{h}']:+.2f}" for h in horizons)
            print(f"{name:<12} {str(r['days'])+'天':<10} {r['rows']:>9,} {r['symbols']:>6}  {cells}")
        except Exception as e:  # 单因子失败不阻塞整批
            print(f"{name:<12} 失败: {e}")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    run_all()
