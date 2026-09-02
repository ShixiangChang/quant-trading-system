# -*- coding: utf-8 -*-
"""纸面交易配置。所有口径尽量对齐回测（model/robust.py）。"""
from pathlib import Path

try:
    from model import config as mcfg
    UNIVERSE = mcfg.UNIVERSE
    PROXY = mcfg.PROXY
    BASE_URL = mcfg.BASE_URL
    INTERVAL = mcfg.INTERVAL
except Exception:  # model 包不可用时兜底
    UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
                "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "TRXUSDT",
                "ETCUSDT", "FILUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "ARBUSDT"]
    PROXY = "http://127.0.0.1:7890"
    BASE_URL = "https://fapi.binance.com"
    INTERVAL = "1h"

PARENT = Path(__file__).resolve().parent.parent
STATE_PATH = PARENT / "data" / "paper_state.json"
LEDGER_PATH = PARENT / "data" / "paper_ledger.csv"
LIVE_IC_PATH = PARENT / "data" / "paper_live_ic.csv"   # 滚动 live IC：每期调仓时的截面 rank-IC 流水

LOOKBACK_BARS = 72      # 拉最近 N 根 1h K 线（48h 高点 + 24h 波动绰绰有余）
REBALANCE_H = 96        # 每批持仓 96h（对齐 96h 截面 z 信号，回测最优点）
BATCH_INTERVAL_H = 8    # 错峰间隔：每 8h 出一批新信号（96h 内 12 批并存，验证密度 ×12）
BATCH_FRAC = BATCH_INTERVAL_H / REBALANCE_H  # 每批资金占比 = 8/96 = 1/12
TOP_PCT = 0.2           # 截面排名前 / 后 20% 做多 / 做空
SLIPPAGE = 0.0006       # 单边成本：手续费 0.04% + 滑点 0.02%（对齐回测 COST_SIDE）
FUNDING = True          # 计资金费（实盘里这是真实费用，必须算）