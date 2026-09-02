from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks

BASE_URL = "https://fapi.binance.com"
TOP_N = 20
MIN_QUOTE_VOLUME = 5_000_000
KLINE_LIMIT = 250
REQUEST_TIMEOUT = 15

TIMEFRAMES = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

class BinanceAPIError(RuntimeError):
    pass

def get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(
        f"{BASE_URL}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and data.get("code", 0) < 0:
        raise BinanceAPIError(str(data))

    return data

def get_valid_symbols() -> set[str]:
    data = get_json("/fapi/v1/exchangeInfo")
    return {
        item["symbol"]
        for item in data["symbols"]
        if item["status"] == "TRADING"
        and item["contractType"] == "PERPETUAL"
        and item["quoteAsset"] == "USDT"
    }

def get_top_gainers() -> pd.DataFrame:
    valid_symbols = get_valid_symbols()
    tickers = get_json("/fapi/v1/ticker/24hr")
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        symbol = ticker["symbol"]
        if symbol not in valid_symbols:
            continue

        change = float(ticker["priceChangePercent"])
        quote_volume = float(ticker["quoteVolume"])

        if change <= 0 or quote_volume < MIN_QUOTE_VOLUME:
            continue

        rows.append(
            {
                "symbol": symbol,
                "change_24h_percent": change,
                "last_price": float(ticker["lastPrice"]),
                "quote_volume_24h": quote_volume,
                "trade_count": int(ticker["count"]),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("change_24h_percent", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )

def get_klines(symbol: str, interval: str) -> pd.DataFrame:
    data = get_json(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": KLINE_LIMIT},
    )

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]

    frame = pd.DataFrame(data, columns=columns)
    numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return frame

def calculate_atr(frame: pd.DataFrame, period: int = 14) -> float:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = true_range.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else float(true_range.mean())

def calculate_adx(frame: pd.DataFrame, period: int = 14) -> float:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    value = dx.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else 0.0

def detect_pivots(frame: pd.DataFrame, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    highs, _ = find_peaks(frame["high"].to_numpy(), distance=order)
    lows, _ = find_peaks(-frame["low"].to_numpy(), distance=order)
    return highs, lows

def cluster_levels(
    prices: list[float],
    current_price: float,
    atr: float,
    tolerance_atr: float = 0.6,
) -> list[dict[str, float]]:
    if not prices:
        return []

    tolerance = max(atr * tolerance_atr, current_price * 0.002)
    clusters: list[list[float]] = []

    for price in sorted(prices):
        if not clusters or price - np.mean(clusters[-1]) > tolerance:
            clusters.append([price])
        else:
            clusters[-1].append(price)

    return [
        {
            "center": float(np.mean(cluster)),
            "lower": float(min(cluster) - tolerance / 2),
            "upper": float(max(cluster) + tolerance / 2),
            "touches": float(len(cluster)),
        }
        for cluster in clusters
    ]

def nearest_zones(
    frame: pd.DataFrame,
    current_price: float,
    atr: float,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    pivot_highs, pivot_lows = detect_pivots(frame)
    resistance_prices = frame.iloc[pivot_highs]["high"].tolist()
    support_prices = frame.iloc[pivot_lows]["low"].tolist()

    resistance_prices += frame["high"].tail(30).nlargest(3).tolist()
    support_prices += frame["low"].tail(30).nsmallest(3).tolist()

    supports = cluster_levels(support_prices, current_price, atr)
    resistances = cluster_levels(resistance_prices, current_price, atr)

    supports = [z for z in supports if z["center"] <= current_price * 1.01]
    resistances = [z for z in resistances if z["center"] >= current_price * 0.99]

    support = max(supports, key=lambda z: z["center"], default=None)
    resistance = min(resistances, key=lambda z: z["center"], default=None)
    return support, resistance

def calculate_channel(frame: pd.DataFrame, atr: float) -> dict[str, Any]:
    closes = frame["close"].tail(80).to_numpy()
    x = np.arange(len(closes), dtype=float)
    slope, intercept = np.polyfit(x, closes, 1)
    fitted = slope * x + intercept
    residual_std = float(np.std(closes - fitted))
    current = float(closes[-1])

    width = max(residual_std * 2.0, atr * 0.8)
    middle = float(fitted[-1])
    slope_percent = slope / current * 100 * len(closes)

    if abs(slope_percent) < 1.0 or residual_std > atr * 2.5:
        channel_type = "horizontal"
    elif slope > 0:
        channel_type = "ascending"
    else:
        channel_type = "descending"

    return {
        "channel_type": channel_type,
        "channel_lower": middle - width,
        "channel_middle": middle,
        "channel_upper": middle + width,
        "channel_slope_percent": float(slope_percent),
    }

def timeframe_bias(frames: dict[str, pd.DataFrame]) -> str:
    directions = []
    for name in ("1h", "4h", "1d"):
        closes = frames[name]["close"].tail(30).to_numpy()
        slope = np.polyfit(np.arange(len(closes)), closes, 1)[0]
        directions.append("up" if slope > 0 else "down")

    if all(x == "up" for x in directions):
        return "bullish"
    if all(x == "down" for x in directions):
        return "bearish"
    return "conflict"

def percent_distance(price: float, level: float | None) -> float | None:
    if level is None or level == 0:
        return None
    return abs(price - level) / price * 100

def determine_breakout(
    price: float,
    frame: pd.DataFrame,
    support: dict[str, float] | None,
    resistance: dict[str, float] | None,
    atr: float,
) -> tuple[str, bool]:
    average_volume = frame["volume"].tail(20).mean()
    volume_confirmed = float(frame["volume"].iloc[-1]) > average_volume * 1.3
    buffer = atr * 0.15

    if resistance and price > resistance["upper"] + buffer:
        return (
            "breakout_up_confirmed" if volume_confirmed else "breakout_up_unconfirmed",
            volume_confirmed,
        )

    if support and price < support["lower"] - buffer:
        return (
            "breakdown_confirmed" if volume_confirmed else "breakdown_unconfirmed",
            volume_confirmed,
        )

    return "inside_structure", volume_confirmed

def signal_result(
    signal: str,
    strategy: str,
    entry: str,
    stop_loss: str,
    take_profit: str,
    risk: str,
    conclusion: str,
) -> dict[str, str]:
    return {
        "signal": signal,
        "strategy": strategy,
        "entry_condition": entry,
        "stop_loss_reference": stop_loss,
        "take_profit_reference": take_profit,
        "risk_level": risk,
        "conclusion": conclusion,
    }

def generate_strategy(row: dict[str, Any]) -> dict[str, str]:
    status = row["breakout_status"]
    trend = row["trend"]
    channel = row["channel_type"]
    bias = row["multi_timeframe_bias"]
    atr = row["atr_1h"]
    daily_change = row["change_24h_percent"]
    distance_support = row["distance_to_support_percent"]
    distance_resistance = row["distance_to_resistance_percent"]

    near_support = distance_support is not None and distance_support <= 1.5
    near_resistance = distance_resistance is not None and distance_resistance <= 1.5

    if status == "breakout_up_confirmed":
        return signal_result(
            "LONG", "突破做多",
            "阻力上方收盘且成交量高于 20 根均量的 1.3 倍",
            f"突破位下方约 {atr:.8g}", "下一阻力区域或上升航道上轨", "medium",
            "做多：阻力位放量突破，等待回踩突破位确认后顺势做多；止损参考突破位下方，止盈参考下一阻力位或上轨",
        )

    if status == "breakout_up_unconfirmed":
        return signal_result(
            "WAIT", "突破未确认",
            "等待成交量放大或后续 K 线继续站稳阻力上方",
            "不适用", "不适用", "high",
            "观望：价格突破阻力但成交量未确认，暂不追入，防范假突破",
        )

    if status == "breakdown_confirmed":
        return signal_result(
            "SHORT", "跌破做空",
            "支撑下方收盘且成交量高于 20 根均量的 1.3 倍",
            f"跌破位上方约 {atr:.8g}", "下一支撑区域或下降航道下轨", "medium",
            "做空：支撑位放量跌破，等待反抽支撑失败后顺势做空；止损参考跌破位上方，止盈参考下一支撑位或下轨",
        )

    if status == "breakdown_unconfirmed":
        return signal_result(
            "WAIT", "跌破未确认",
            "等待收盘确认和成交量放大",
            "不适用", "不适用", "high",
            "观望：价格跌破支撑但确认不足，暂不追空，防范假跌破",
        )

    if bias == "conflict":
        return signal_result(
            "WAIT", "多周期冲突",
            "等待 1h、4h、1d 方向统一",
            "不适用", "不适用", "high",
            "观望：多周期方向冲突，当前做多和做空都缺乏明确优势",
        )

    if trend == "uptrend" and bias == "bullish" and near_support:
        return signal_result(
            "LONG", "回踩支撑做多",
            "回踩支撑后出现止跌、阳线反包或低点抬高",
            f"支撑区域下方约 {atr:.8g}", "前方阻力区域或上升航道上轨", "medium",
            "做多：大周期和当前趋势一致向上，价格回踩支撑；等待止跌确认后做多，止损放在支撑下方",
        )

    if trend == "downtrend" and bias == "bearish" and near_resistance:
        return signal_result(
            "SHORT", "反抽阻力做空",
            "反抽阻力后出现冲高回落、阴线反包或高点降低",
            f"阻力区域上方约 {atr:.8g}", "前方支撑区域或下降航道下轨", "medium",
            "做空：大周期和当前趋势一致向下，价格反抽阻力；等待受阻确认后做空，止损放在阻力上方",
        )

    if channel == "horizontal" and near_support:
        return signal_result(
            "LONG", "震荡下沿做多",
            "接近区间下沿并出现止跌确认，未有效跌破支撑",
            f"区间下沿下方约 {atr:.8g}", "区间中轨或上沿", "medium",
            "做多：价格接近震荡区间下沿，等待止跌确认后轻仓做多；止盈参考中轨或区间上沿",
        )

    if channel == "horizontal" and near_resistance:
        return signal_result(
            "SHORT", "震荡上沿做空",
            "接近区间上沿并出现冲高回落，未有效突破阻力",
            f"区间上沿上方约 {atr:.8g}", "区间中轨或下沿", "medium",
            "做空：价格接近震荡区间上沿，等待受阻确认后轻仓做空；止盈参考中轨或区间下沿",
        )

    if trend == "uptrend" and daily_change >= 20:
        return signal_result(
            "WAIT", "上涨过热观望",
            "等待回踩支撑或放量突破后重新评估",
            "不适用", "不适用", "high",
            "观望：上涨趋势仍在但短线涨幅过大，不追多，也不在没有反转确认时盲目做空",
        )

    if trend == "downtrend":
        return signal_result(
            "WAIT", "下降趋势观望",
            "等待反抽阻力失败后做空，或等待结构反转后再做多",
            "不适用", "不适用", "high",
            "观望：处于下降趋势但当前位置不满足高盈亏比做空条件，等待反抽阻力或新结构形成",
        )

    return signal_result(
        "WAIT", "区间中部观望",
        "等待接近支撑、阻力，或出现有效突破/跌破",
        "不适用", "不适用", "low",
        "观望：价格位于结构中部，当前盈亏比不足，不主动开仓",
    )

def analyze_symbol(symbol: str, change: float, quote_volume: float) -> dict[str, Any]:
    frames = {name: get_klines(symbol, interval) for name, interval in TIMEFRAMES.items()}
    frame = frames["1h"]
    price = float(frame["close"].iloc[-1])
    atr = calculate_atr(frame)
    adx = calculate_adx(frame)
    channel = calculate_channel(frame, atr)
    support, resistance = nearest_zones(frame, price, atr)

    slope = channel["channel_slope_percent"]
    if channel["channel_type"] == "ascending" or (adx >= 25 and slope > 0):
        trend = "uptrend"
    elif channel["channel_type"] == "descending" or (adx >= 25 and slope < 0):
        trend = "downtrend"
    else:
        trend = "sideways"

    breakout_status, volume_confirmed = determine_breakout(
        price, frame, support, resistance, atr
    )

    row: dict[str, Any] = {
        "symbol": symbol,
        "change_24h_percent": change,
        "quote_volume_24h": quote_volume,
        "current_price": price,
        "atr_1h": atr,
        "adx_1h": adx,
        "trend": trend,
        **channel,
        "support_zone": f"{support['lower']:.8g} - {support['upper']:.8g}" if support else "N/A",
        "resistance_zone": f"{resistance['lower']:.8g} - {resistance['upper']:.8g}" if resistance else "N/A",
        "support_center": support["center"] if support else None,
        "resistance_center": resistance["center"] if resistance else None,
        "distance_to_support_percent": percent_distance(price, support["upper"] if support else None),
        "distance_to_resistance_percent": percent_distance(price, resistance["lower"] if resistance else None),
        "breakout_status": breakout_status,
        "volume_confirmed": volume_confirmed,
        "multi_timeframe_bias": timeframe_bias(frames),
    }

    row["market_status"] = f"{trend}, {channel['channel_type']} channel, {breakout_status}"
    row.update(generate_strategy(row))
    return row

def main() -> None:
    print("正在获取 Binance U 本位永续合约涨幅榜...")
    gainers = get_top_gainers()

    if gainers.empty:
        print("没有找到符合条件的上涨交易对。")
        return

    results: list[dict[str, Any]] = []

    for index, item in gainers.iterrows():
        symbol = item["symbol"]
        print(f"正在分析 {index + 1}/{len(gainers)}: {symbol}")
        try:
            results.append(
                analyze_symbol(
                    symbol,
                    float(item["change_24h_percent"]),
                    float(item["quote_volume_24h"]),
                )
            )
        except Exception as exc:
            print(f"跳过 {symbol}，分析失败: {exc}")
        time.sleep(0.15)

    if not results:
        print("没有成功完成任何交易对分析。")
        return

    result_frame = pd.DataFrame(results)
    result_frame.insert(0, "rank", range(1, len(result_frame) + 1))
    result_frame.to_csv("binance_market_analysis.csv", index=False, encoding="utf-8-sig")

    display_columns = [
        "rank", "symbol", "change_24h_percent", "current_price", "trend",
        "channel_type", "support_zone", "resistance_zone", "breakout_status",
        "multi_timeframe_bias", "signal", "strategy", "risk_level",
        "entry_condition", "stop_loss_reference", "take_profit_reference",
        "conclusion",
    ]

    output = result_frame[display_columns].copy()
    output["change_24h_percent"] = output["change_24h_percent"].map(lambda x: f"{x:.2f}%")
    output["current_price"] = output["current_price"].map(lambda x: f"{x:.8g}")

    print("\n分析结果：\n")
    print(output.to_string(index=False))
    print("\n完整结果已保存到 binance_market_analysis.csv")

if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"网络请求失败: {exc}")
    except (BinanceAPIError, KeyError, ValueError) as exc:
        print(f"程序执行失败: {exc}")