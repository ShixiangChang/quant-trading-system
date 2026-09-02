# -*- coding: utf-8 -*-
"""建模配置：所有可调参数集中在这里。"""
from pathlib import Path

# ---------------------------------------------------------------- 数据
BASE_URL = "https://fapi.binance.com"
PROXY = "http://127.0.0.1:7890"   # 直连不通时走 Clash 代理（默认端口 7890）
INTERVAL = "1h"               # K 线周期（特征与预测都在此周期）
DAYS = 730                    # 拉取历史天数（折数更多，验证更可信）
KLINE_LIMIT = 1500            # 单次 K 线请求上限（Binance 限制）

# 交易对池：老牌流动性好的 USDT 永续（历史深）；新币种历史短，数据层按实有长度处理
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "TRXUSDT", "ETCUSDT", "FILUSDT", "UNIUSDT",
    "ATOMUSDT", "NEARUSDT", "ARBUSDT",
]

# ---------------------------------------------------------------- 缓存
CACHE_DIR = Path(Path(__file__).resolve().parent.parent / "data" / "model_cache")

# ---------------------------------------------------------------- 标签 / 预测
HORIZON = 4                # 主预测周期（向后兼容旧代码）
HORIZONS = [1, 4, 8, 12, 24, 48, 72, 96, 168]   # sweep 网格搜索时尝试的预测周期
MIN_HISTORY_BARS = 300     # 特征所需最少 K 线，不足的币种丢弃
USE_FUNDING = True         # 是否用资金费率为特征（深历史已支持）
USE_MICRO = True           # 是否用 OI/多空比/主动买卖 微观结构特征（monitor 已实时验证有信号）；回测闸门传 use_micro=False 跳过

# ---------------------------------------------------------------- 回测 / 验证
COST_SIDE = 0.0006         # 单边成本：手续费 0.04% + 滑点 0.02%
SIGNAL_Z = 1.0             # 截面 z 阈值：pred_z >= +SIGNAL_Z 做多，<= -SIGNAL_Z 做空，|z|< 阈值空仓
TRAIN_DAYS = 90            # 每折训练窗口
TEST_DAYS = 30             # 每折测试窗口
STEP_DAYS = 30             # 滚动步进
MIN_TRAIN_ROWS = 5000      # 训练样本不足则跳过该折

# ---------------------------------------------------------------- LightGBM
# qlib LightGBM benchmark 同款：大模型 + 极强正则（反直觉，但工业界验证过）。
# 大 leaves 容易过拟合，靠 lambda_l1/l2 强正则兜住，配高 lr（不用早停也能收敛）。
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.2,
    "num_leaves": 210,
    "max_depth": 8,
    "lambda_l1": 205.7,
    "lambda_l2": 581.0,
    "feature_fraction": 0.8879,   # = colsample_bytree
    "bagging_fraction": 0.8789,   # = subsample
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 400
ENSEMBLE_SEEDS = 5        # 单折内集成多少颗不同 seed 的树取平均预测，压掉 big-model 的训练方差（seed 敏感）

# ---------------------------------------------------------------- 输出
OUTPUT_DIR = Path(Path(__file__).resolve().parent.parent / "data" / "model_out")