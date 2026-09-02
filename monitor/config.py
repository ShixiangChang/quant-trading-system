# -*- coding: utf-8 -*-
"""全局配置：所有可调参数集中在这里，改参数只需要动这一个文件。"""
from pathlib import Path

# ---------------------------------------------------------------- 网络
BASE_URL = "https://fapi.binance.com"
WS_URL = "wss://fstream.binance.com/stream"
# 如果直连 Binance 不通（国内网络通常需要），填代理，例如 "http://127.0.0.1:7890"
PROXY = "http://127.0.0.1:7890"
TIMEOUT = 15

# ---------------------------------------------------------------- 告警渠道
# "feishu" = 飞书 | "dingtalk" = 钉钉
NOTIFY_CHANNEL = "dingtalk"
# 真实 webhook / secret 只留本机 monitor/secrets.py（已 gitignore，不进版本库）。
# 这里从 secrets 读取；secrets 缺失或字段没填时兜底为空（== 告警只能打印到控制台）。
try:
    from . import secrets as _sec
    FEISHU_WEBHOOK = getattr(_sec, "FEISHU_WEBHOOK", "")
    DINGTALK_WEBHOOK = getattr(_sec, "DINGTALK_WEBHOOK", "")
    DINGTALK_SECRET = getattr(_sec, "DINGTALK_SECRET", "")
    DINGTALK_DECISION_WEBHOOK = getattr(_sec, "DINGTALK_DECISION_WEBHOOK", "")
    DINGTALK_DECISION_SECRET = getattr(_sec, "DINGTALK_DECISION_SECRET", "")
except ImportError:
    FEISHU_WEBHOOK = ""
    DINGTALK_WEBHOOK = ""
    DINGTALK_SECRET = ""
    DINGTALK_DECISION_WEBHOOK = ""
    DINGTALK_DECISION_SECRET = ""
# True = 只把告警打印到控制台，不发送 webhook（配好 webhook 后改成 False）
# 2026-08-29：核心目的转为积累数据建模，告警推送关闭（落库照常）
DRY_RUN = True

# ---------------------------------------------------------------- 监控池
# 始终订阅 24h 成交额前 N 名（基准池），保证基础数据流
BASELINE_TOP_N = 10
BASELINE_REFRESH_SEC = 900          # 每 15 分钟刷新一次基准池和涨幅榜
WATCHLIST_TTL_SEC = 6 * 3600        # 触发币种在 6 小时无新事件后移出深度监控
# 分层扫描：L0 全市场 → L1 观察池 → L2 深度池
LOOKOUT_MAX = 30                    # 观察池上限（防全市场深度订阅爆炸）
DEPTH_TOP_N = 0                     # 深度池大小：观察池里最热的 N 个币才订阅盘口（建模无盘口特征，已关停）

# 异动扫描进观察池的门槛（涨跌对称：|24h 涨跌幅| 分桶 + 24h 成交额分桶）
# 成交额越大门槛越低，长尾币也能进池；暴涨和暴跌同样触发。
ENTRY_TIERS = [
    (100_000_000, 15.0),   # 24h 成交额 >= 1 亿：|涨跌幅| >= 15%
    (20_000_000, 20.0),    # >= 2000 万：|涨跌幅| >= 20%
    (0, 25.0),             # 其余长尾：|涨跌幅| >= 25%
]

# ---------------------------------------------------------------- 事件阈值
# ---- 归一化原则：异动 = 相对该币自身常态，一律不用绝对美元当门槛 ----
# 「100 万美元对 BTC 是尘埃、对山寨币是地震」，绝对金额门槛是外行口径。

# 大单（只落明细库；告警改用下方「资金流净流入」的归一化信号）
LARGE_TRADE_DB_USD = 100_000        # 单笔 >= 10 万美元写入明细库（建模原料）

# 资金流净流入（相对自身成交额的吸筹/派发，跨币种可比，替代单笔大单告警）
FLOW_WINDOW_SEC = 300               # 聚合窗口 5 分钟
FLOW_RATIO_THRESH = 0.30            # |净主动买入额/窗口成交额| >= 30% 视为异常
FLOW_CONSECUTIVE = 2                # 连续 N 个窗口同向才算持续（压单窗口噪声）
FLOW_TOTAL_MIN_RATIO = 1.0          # 窗口成交额 >= 该币 24h 均量(5m)×此倍数，防冷清币几笔刷高占比
FLOW_COOLDOWN_SEC = 1800            # 同币同方向告警冷却 30 分钟

# 爆仓潮（窗口内累计，相对该币自身 OI，跨币种可比）
LIQ_WINDOW_SEC = 300                # 统计窗口 5 分钟
LIQ_OI_RATIO = 0.05                 # 窗口累计爆仓额 >= 该币当前 OI 的 5%（替代 100 万绝对额）
LIQ_COUNT = 10                      # 或窗口内 >= 10 笔（新币暂无 OI 时兜底）
LIQ_COOLDOWN_SEC = 900              # 同币种爆仓告警冷却 15 分钟

# 持仓量 OI 异动（相对自身变化率，不设绝对金额门槛）
OI_WINDOW_SEC = 300                 # 对比 5 分钟前的 OI
OI_CHANGE_PCT = 3.0                 # 变化率 >= 3%
OI_COOLDOWN_SEC = 1800              # 冷却 30 分钟

# 盘口失衡（L2 深度池，报突破前兆）。原始盘口即用即弃，只落聚合快照。
DEPTH_LEVELS = 20                   # 盘口档位（bids/asks 各 20 档）
DEPTH_IMBALANCE_THRESH = 0.70      # bid_imbalance >= 此值偏多 / <= 1-此值偏空
DEPTH_CONFIRM = 3                   # 连续 N 次快照同向才算，压畸形盘口噪声
DEPTH_POLL_SEC = 10                 # REST 轮询盘口间隔（仅对 DEPTH_TOP_N 个币）
DEPTH_COOLDOWN_SEC = 1800           # 同币同方向盘口告警冷却 30 分钟

# 资金费率极值
FUNDING_EXTREME = 0.0005            # |费率| >= 0.05%/8h（基准值 0.01%）
FUNDING_COOLDOWN_SEC = 14400        # 冷却 4 小时

# ---------------------------------------------------------------- REST 轮询周期
OI_POLL_SEC = 30                    # 持仓量
RATIO_POLL_SEC = 60                 # 多空账户比 / 大户持仓比 / 主动买卖比

# ---------------------------------------------------------------- WebSocket
WS_STALE_SEC = 90                   # WS 超过 N 秒无数据 → 看门狗启动 REST 降级兜底

# ---------------------------------------------------------------- 存储
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "monitor.db")

# ---------------------------------------------------------------- 数据健康哨兵
# 定时核检 monitor.db：发现断档/脏值立刻告警，每天一条健康心跳（沉默 = 确认干净）
HEALTH_CHECK_SEC = 900             # 每 15 分钟核检一次
HEALTH_STALE_SEC = 3 * 3600        # 最后写入超过 3 小时算断档（oi/ratios/mark_prices 正常应分钟级更新）
HEALTH_MIN_SYMBOLS = 1             # oi 近 24h 覆盖币种数低于此值告警。监控池动态，先设保守值 1（扩池后收紧）
HEALTH_HEARTBEAT_SEC = 24 * 3600   # 每天一条健康日报（各表行数 + 新鲜度）

# ---------------------------------------------------------------- 数据可靠性（工业级五道闸门）
HEARTBEAT_SEC = 60                  # 独立心跳进程写入间隔（秒）
EXTERNAL_PING_URL = ""              # 外部存活检测 URL（UptimeRobot 等心跳端点）；空 = 只写库不对外 ping

# ---------------------------------------------------------------- 链上监控
# 总开关：2026-08-29 曾因「建模无链上特征」关停，2026-09-01 恢复（现货链上数据是
# 现货链上数据需持续落库积累，即使短期不进模型）
ONCHAIN_ENABLED = True
# Etherscan 已迁移到 V2 接口（/v2/api + chainid 参数）。
# 注意：Etherscan 与 BSCScan 是两套独立 key：
#   - ETHERSCAN_API_KEY 只覆盖以太坊主网（chainid=1）；免费档访问不了其它链。
#   - BSC 需要去 https://bscscan.com 单独注册一个免费 key 填到 BSCSCAN_API_KEY。
# 密钥不硬编码在这里（避免进 Git）。真实值放在同目录 secrets.py（已 gitignore），新机器克隆后自建。
try:
    from .secrets import ETHERSCAN_API_KEY, BSCSCAN_API_KEY
except ImportError:
    ETHERSCAN_API_KEY = ""
    BSCSCAN_API_KEY = ""
# 链配置：域名 + 链 id + 对应 key（没填 key 的链自动跳过）
ONCHAIN_CHAINS = {
    "eth": {"url": "https://api.etherscan.io", "chainid": 1, "key": ETHERSCAN_API_KEY},
    "bsc": {"url": "https://api.bscscan.com", "chainid": 56, "key": BSCSCAN_API_KEY},
}
# 鲸鱼转账阈值
WHALE_TRANSFER_USD = 5_000_000      # 单笔 >= 500 万美元才告警
WHALE_TRANSFER_DB_USD = 1_000_000   # >= 100 万美元写入数据库（给 建模）
WHALE_COOLDOWN_SEC = 1800           # 同链同 token 告警冷却 30 分钟
ONCHAIN_POLL_SEC = 10               # 轮询间隔（免费档 5 次/秒，很宽裕）
ONCHAIN_LATEST_N = 1000             # 每次拉最新 N 条转账

# 动态层：全链扫描主流稳定币最新大额转账（无需预设地址，自动发现鲸鱼）
STABLE_TOKENS = {
    "eth": {
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    },
    "bsc": {
        "USDT": "0x55d398326f99059fF775485246999027B3197955",
        "BUSD": "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",
    },
}

# 现货链上数据扩展（2026-09-01）：ERC-20 代币交易所净流入流出。
# 背景：Etherscan tokenholderlist（持仓集中度）是 API Pro 端点，免费档拿不到；
# 免费可行的替代是 tokentx 接口监控代币本身在交易所的净流入流出
# （代币流入交易所=潜在抛压，流出=囤币/提现离场），即现货 netflow 的散户版。
# 慢变量，由 monitor/erc20_flow.py 定时跑，落 erc20_flow 表。
ERC20_TOKENS = {
    "eth": {
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    },
}
# 代币净流入流出的最小美元门槛（低于此值的转账不落库，压噪声）
ERC20_FLOW_MIN_USD = 100_000
# ERC-20 净流入采集轮询间隔（秒）。慢变量，10 分钟一轮足够，免费档限速也扛得住。
ERC20_FLOW_POLL_SEC = 600

# 交易所热钱包地址（公开），用于识别 ERC-20 代币流入/流出交易所。
# 流入交易所=潜在抛压，流出=囤币/提现离场。地址会变，需定期维护补充。
EXCHANGE_ADDRS = {
    "eth": {
        "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
        "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "Kraken",
    },
    "bsc": {
        "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
    },
}

# 固定层：已知鲸鱼钱包。role: exchange=交易所热钱包 / treasury=稳定币金库。
# 大额流入交易所 = 潜在抛压；流出 = 囤币离场。地址是公开的，可随时增删。
WHALE_WALLETS = {
    "Binance 热钱包": {
        "role": "exchange",
        "eth": "0x28C6c06298d514Db089934071355E5743bf21d60",
        "bsc": "0x28C6c06298d514Db089934071355E5743bf21d60",
    },
    "Tether 金库": {
        "role": "treasury",
        "eth": "0x5754284f345afc66a98fbB0a0Afe71e0F007B949",
        "bsc": "",
    },
}
