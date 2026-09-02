# -*- coding: utf-8 -*-
"""市值/流动性数据源：分层体系（规模=市值，可交易性=流动性/换手率，均为连续量）。

废弃旧的「24h 成交额 ≥ 1 亿」二元硬切。改为三层专业标准：
  1) 规模     = 市值（存量）。CoinGecko 拉全市场市值，24h 缓存（慢变量）。
  2) 可交易性 = 流动性门槛：成交额 ≥ MIN_VOLUME 且 换手率（成交额÷市值）≥ MIN_TURNOVER。
                只回答「能不能交易」，不回答「大不大」。
  3) 连续特征 = 市值、成交额、换手率 → 进风控（仓位缩放）和建模，不做任何二分类。

为什么不再用「成交额切大小」：成交额是流量、市值是存量，两个正交维度。
高市值低成交（锁仓/盘口浅）的币卖不出去，低市值高成交（爆炒 memecoin）短期可交易——
单一指标切分必然把两者都归错。市值定规模、流动性当门槛，各干各的。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from . import config as mcfg

_CACHE_FILE = mcfg.CACHE_DIR / "market_cap.json"
_CACHE_TTL = 24 * 3600          # 市值是慢变量，每天刷新一次即可

# —— 分层阈值（连续门槛，非二元标签）——
MCAP_TOP_PCT = 0.30             # 池内市值分位数 top 30% 算「大」规模
MIN_VOLUME = 5_000_000          # 流动性硬门槛：24h 成交额 ≥ $5M
MIN_TURNOVER = 0.005            # 换手率门槛：成交额 ÷ 市值 ≥ 0.5%（剔除死币/刷量币）


def _fetch_coingecko_market_caps() -> dict:
    """CoinGecko 拉全市场市值，返回 {coin_code_upper: mcap}。symbol 撞名时取市值最大者。"""
    proxies = {"http": mcfg.PROXY, "https": mcfg.PROXY} if mcfg.PROXY else None
    out: dict[str, float] = {}
    for page in (1, 2, 3):      # 250×3 = 750 币，覆盖池子绰绰有余
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={"vs_currency": "usd", "order": "market_cap_desc",
                        "per_page": 250, "page": page},
                proxies=proxies, timeout=20,
            )
            if r.status_code != 200:
                break
            for d in r.json():
                sym = str(d.get("symbol", "")).upper()
                mcap = float(d.get("market_cap") or 0.0)
                if sym and mcap > out.get(sym, 0.0):   # 撞名取市值最大（主流币）
                    out[sym] = mcap
        except requests.RequestException:
            break
    return out


def _mcap_cached(force: bool = False) -> dict:
    """市值缓存（24h TTL），失败回退磁盘、再失败返回空。"""
    if not force and _CACHE_FILE.exists():
        try:
            c = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - c.get("ts", 0) < _CACHE_TTL:
                return c.get("mcap", {})
        except Exception:
            pass
    mcap = _fetch_coingecko_market_caps()
    if mcap:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps({"ts": time.time(), "mcap": mcap}),
                                   encoding="utf-8")
        except Exception:
            pass
    return mcap


def _coin_of(symbol: str) -> str:
    """Binance 永续 symbol → 币代码：BTCUSDT→BTC，1000PEPEUSDT→PEPE。"""
    c = symbol.replace("USDT", "").replace("USDC", "")
    if c.startswith("1000"):
        c = c[4:]
    return c.upper()


def market_profile(symbols, force: bool = False) -> dict:
    """给一批 Binance symbol 返回 {symbol: {mcap, volume, turnover}}。

    mcap=市值（CoinGecko，24h 缓存）；volume=24h 成交额（Binance 实时）；
    turnover=换手率 = volume/mcap（<0 或 mcap 缺失时记 0）。
    """
    mcap_map = _mcap_cached(force)
    try:
        from model.data import _get_json
        tick = _get_json("/fapi/v1/ticker/24hr")
        vol_map = {d["symbol"]: float(d.get("quoteVolume", 0)) for d in tick}
    except Exception:
        vol_map = {}

    profile: dict = {}
    for sym in symbols:
        coin = _coin_of(sym)
        mcap = mcap_map.get(coin, 0.0)
        vol = vol_map.get(sym, 0.0)
        turnover = (vol / mcap) if mcap > 0 else 0.0
        profile[sym] = {"mcap": mcap, "volume": vol, "turnover": turnover}
    return profile


def large_cap_symbols(symbols, profile: dict, top_pct: float = MCAP_TOP_PCT) -> set:
    """规模分层：池内市值分位数 top_pct 的算「大」。只回答规模，不含流动性。"""
    mcaps = [(s, profile.get(s, {}).get("mcap", 0.0)) for s in symbols]
    ranked = sorted([m for _, m in mcaps if m > 0], reverse=True)
    if not ranked:
        return set()
    k = max(1, int(len(ranked) * top_pct))
    threshold = ranked[k - 1] if k <= len(ranked) else ranked[-1]
    return {s for s, m in mcaps if m > 0 and m >= threshold}


def tradeable_symbols(profile: dict,
                      min_volume: float = MIN_VOLUME,
                      min_turnover: float = MIN_TURNOVER) -> set:
    """流动性硬门槛：成交额 ≥ min_volume 且 换手率 ≥ min_turnover，才能进可交易池。"""
    out = set()
    for sym, p in profile.items():
        if p.get("volume", 0) >= min_volume and p.get("turnover", 0) >= min_turnover:
            out.add(sym)
    return out


def liquidity_scale(turnover: float, ref: float = 0.05, floor: float = 0.3) -> float:
    """流动性 → 仓位缩放因子（连续，非二元）：换手率 ref 以上全仓，以下线性缩到 floor。"""
    t = float(turnover or 0.0)
    if t <= 0:
        return floor
    return min(max(t / ref, floor), 1.0)
