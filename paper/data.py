# -*- coding: utf-8 -*-
"""纸面交易数据层：轻量拉取最近 K 线 + 资金费（不碰模型的重缓存，保证拿到当前价）。"""
from __future__ import annotations

import requests
import pandas as pd

from . import config


def _get(path: str, params: dict):
    proxies = {"http": config.PROXY, "https": config.PROXY} if config.PROXY else None
    r = requests.get(config.BASE_URL + path, params=params, timeout=20, proxies=proxies)
    r.raise_for_status()
    return r.json()


def fetch_recent_klines(symbol: str, limit: int | None = None) -> pd.DataFrame:
    """最近 N 根 1h K 线（含 high/low/close）。"""
    limit = limit or config.LOOKBACK_BARS
    data = _get("/fapi/v1/klines", {"symbol": symbol, "interval": config.INTERVAL, "limit": limit})
    rows = [{"open_time": int(k[0]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4])}
            for k in data]
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def fetch_funding_since(symbol: str, start_ms: int) -> list[dict]:
    """自 start_ms 以来的资金费率记录，用于精确累计持仓期资金费。"""
    return _get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start_ms, "limit": 1000})