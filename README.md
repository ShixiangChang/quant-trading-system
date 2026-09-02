# quant-trading-system

币安 USDT 本位永续合约的量化交易系统：实时行情监控 → 机器学习信号 → 多策略组合决策 → 纸面交易结算。

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-00875A)
![SQLite](https://img.shields.io/badge/Storage-SQLite-003B57?logo=sqlite&logoColor=white)
![Binance](https://img.shields.io/badge/Exchange-Binance%20FAPI-F0B90B)
![License](https://img.shields.io/badge/License-MIT-097eff)

---

## 功能

- **实时监控**：Binance WebSocket/REST 行情，监控大单、爆仓、持仓量异动、资金费率极值，以及链上大额转账，推送飞书/钉钉告警并落库。
- **机器学习信号**：LightGBM 双目标回归（截面 z 与波动率调整），walk-forward 样本外验证。
- **多策略组合**：五条独立结算的策略腿，各腿独立跟踪净值。
- **纸面交易**：多腿独立账户结算 + 插针抄底独立账户，决策快照落库可复盘。
- **本地看板**：净值、对账、回放对比的 HTTP 看板。

---

## 系统架构

```
                        ┌──────────────────────────────────┐
  Binance FAPI          │            monitor/              │
  WS + REST ───────────▶│  实时监控：大单 / 爆仓 / OI / 费率│──▶ 飞书 / 钉钉
  Etherscan / BSCScan ─▶│  链上鲸鱼：稳定币大额转账 / 交易所 │──▶ 告警
                        └────────────────┬─────────────────┘
                                         │ monitor.db (SQLite)
                                         ▼
                        ┌──────────────────────────────────┐
                        │              model/              │
                        │  特征工程 + LightGBM 双目标回归    │
                        │  (截面z + 波动率调整) · 多周期     │
                        │  walk-forward 回测 · 验证引擎      │
                        └────────────────┬─────────────────┘
                                         │ pred_z 信号
                                         ▼
                        ┌──────────────────────────────────┐
                        │             decision/            │
                        │  决策机器人：每 8h 出操作清单      │
                        │  多腿组合 · 前向自愈 · 逐日对账    │
                        └────────────────┬─────────────────┘
                                         │ 方向 + 价格 + 仓位 + 理由
                                         ▼
                        ┌──────────────────────────────────┐
                        │              paper/              │
                        │  纸面引擎：多腿独立账户结算净值     │
                        │  + 插针抄底独立账户               │
                        └────────────────┬─────────────────┘
                                         │ 净值 → 淘汰 / 加码
                                         ▼
                                （可选）实盘执行
```

---

## 技术栈

| 层 | 技术 | 用途 |
|---|---|---|
| 行情接入 | Binance WebSocket / REST、Etherscan / BSCScan | 实时行情、链上大额转账 |
| 数据工程 | pandas · numpy · scipy | 特征工程、统计计算 |
| 异步网络 | aiohttp | 并发抓取 75 个交易对 |
| 机器学习 | LightGBM 4.x | 双目标截面回归 |
| 存储 | SQLite | 行情、事件流、决策快照 |
| 告警 | 飞书 / 钉钉 webhook | 实时推送与对账 |

---

## 模块导览

| 模块 | 职责 | 入口 |
|---|---|---|
| `monitor/` | 实时监控：大单、爆仓潮、持仓量异动、资金费率极值、链上鲸鱼 → 推飞书/钉钉并落库 | `python -m monitor.main` |
| `model/` | 特征工程 + LightGBM 双目标回归 + walk-forward 回测 + 验证引擎 | `python -m model.sweep` |
| `decision/` | 决策机器人：每 8h 出一份「方向 + 开/止损/减/加仓价 + 仓位 + 理由」清单 → 推钉钉 | `python -m decision` |
| `paper/` | 纸面引擎：多腿独立账户结算净值 + 插针抄底独立账户 | `python -m paper` / `python -m paper.pin_engine` |
| `backtest/` | 历史回放扫描：验证各腿收益、参数敏感性、组合稳健性 | `python -m backtest.<script>` |
| `status.py` | 本地看板（净值 / 对账 / 回放对比），HTTP 服务 | `status.py serve --port 8777` |

---

## 快速开始

需要 Python 3.13+。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥（模板已给，真实 key 不进版本库）
cp monitor/secrets_example.py monitor/secrets.py   # 填入 ETHERSCAN_API_KEY / BSCSCAN_API_KEY
#    编辑 monitor/config.py 与 model/config.py：
#      - FEISHU_WEBHOOK / DINGTALK_WEBHOOK：填机器人 webhook
#      - PROXY：国内连不上 Binance 时填 "http://127.0.0.1:7890"
#      - DRY_RUN：配好 webhook 后改为 False

# 3. 运行
python -m monitor.main --test      # 测 webhook 是否配好
python -m monitor.main             # 实时监控（常驻）
python -m decision                 # 出今日操作清单（--dry 只打印不推送）
python -m paper                    # 纸面结算 + 换仓 + 推净值
python -m paper.pin_engine         # 插针抄底纸面引擎
status.py serve --port 8777        # 本地看板
```

> 每套常驻进程配有一键启动脚本（`start_monitor.bat` / `start_pin.bat` / `start_status.bat`），
> Windows 下双击即用；脚本假定 `python` 已在 PATH 上。

---

## 策略

五条策略腿各自独立结算净值：

| 腿 | 逻辑 |
|---|---|
| **β 趋势跟随** | 全池等权指数相对 30 日均线的位置决定多空方向，净敞口随行情强弱连续调节 |
| **中小币截面** | 市值 top30% 以下 + 流动性门槛的币种做截面多空 |
| **事件驱动** | 大单 / 资金流 / OI 异动 / 盘口失衡等事件触发微小仓位 |
| **慢动量** | 30 天回看横截面动量，只做多强势币 |
| **插针抄底** | 15min 急跌超阈值做多、持有 12h，仓位按跌幅深度加权 |

---

## 数据与状态

运行时数据落在 `data/`（已 gitignore，不进版本库）：

| 路径 | 内容 |
|---|---|
| `data/monitor.db` | SQLite：K 线、资金费率、事件流、交易台账、决策快照（`decision_snapshots`） |
| `data/model_cache/` `data/model_out/` | 模型缓存与回测输出 |
| `data/paper_state.json` 等 | 各腿纸面净值与持仓状态 |

---

## 目录结构

```
quant-trading-system/
├── monitor/            # 实时监控：WS→REST 降级、数据健康哨兵、链上鲸鱼
├── model/              # 机器学习：特征工程、LightGBM 双目标、walk-forward、验证引擎
├── decision/           # 决策机器人：多周期预测 → 组合 → 操作清单 → 钉钉
├── paper/              # 纸面引擎：多腿独立账户 + 插针抄底
├── backtest/           # 历史回放扫描：各腿收益验证、参数敏感性、组合稳健性
├── status.py           # 本地看板（127.0.0.1:8777）
├── market_analysis.py  # 涨幅榜 Top20 技术面规则分析工具
└── requirements.txt    # aiohttp / pandas / numpy / scipy / lightgbm
```

---

## 隐私与安全

- 真实密钥（webhook / API key）只存在于本机 `monitor/secrets.py`，已被 `.gitignore` 排除，仓库内只有空模板 `secrets_example.py`。
- `data/`（数据库、模型缓存、回测输出）不进版本库。

---

## 许可证

[MIT](./LICENSE) © 2026 ShixiangChang

---

## 免责声明

本项目为研究与交易用途，代码与历史回测结果不构成任何投资建议。加密货币合约交易风险极高（含爆仓归零风险），过去的回测表现不代表未来收益。使用本项目代码进行实盘交易，风险自负。
