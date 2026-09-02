# -*- coding: utf-8 -*-
"""把「插针 + 慢动量 v2」组合落成独立轨道 JSON，供看板 status.py 展示。

组合逻辑：对齐两腿共同交易日，日收益 combo = w*pin_ret + (1-w)*v2_ret，
w 为插针权重（默认 0.70，扫描最优：Sharpe 3.07 / 回撤 -8.3%）。
不覆盖任何旧账户，仅新增一条独立组合轨道。
用法：python -m backtest.combine_track [--w 0.70]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TRACKS_DIR = Path("data/model_out/tracks")
PIN_SRC = "data/model_out/pin_strategy_12h.json"
V2_SRC = "data/model_out/replay_all.json"
W_PIN = 0.70


def _fmt_day(d: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(d * 86400, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def _to_dret(eq: dict[int, float]) -> dict[int, float]:
    days = sorted(eq)
    return {days[i]: eq[days[i]] / eq[days[i - 1]] - 1.0 for i in range(1, len(days))}


def build(w: float) -> None:
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)

    pin_eq = {int(d): float(v) for d, v in json.loads(
        Path(PIN_SRC).read_text(encoding="utf-8"))["equity"]}
    v2_eq = {int(int(x[0]) // 86400): float(x[1]) for x in json.loads(
        Path(V2_SRC).read_text(encoding="utf-8"))["legs"]["v2"]["equity"]}

    pin_ret = _to_dret(pin_eq)
    v2_ret = _to_dret(v2_eq)
    common = sorted(set(pin_ret) & set(v2_ret))

    p = np.array([pin_ret[d] for d in common])
    v = np.array([v2_ret[d] for d in common])
    combo = w * p + (1 - w) * v
    eq = np.cumprod(1 + combo)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    annual = float(eq[-1] ** (365 / len(combo)) - 1)
    sharpe = float(combo.mean() / (combo.std() + 1e-12) * np.sqrt(365))
    total = float(eq[-1] - 1)
    hit = float((combo > 0).mean())

    stats = {
        "total_ret": total,
        "max_drawdown": dd,
        "hit_rate": hit,
        "annual": annual,
        "sharpe": sharpe,
        "events": 0,
        "odds": 0.0,
    }
    out = {
        "name": f"★插针+慢动量组合·{int(w*100)}/{int((1-w)*100)}",
        "start": _fmt_day(common[0]),
        "end": _fmt_day(common[-1]),
        "stats": stats,
        "equity": [[int(d) * 86400, round(float(e), 4)] for d, e in zip(common, eq)],
        "monthly": {},
    }
    dest = TRACKS_DIR / "combo_pin_v2.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[combine_track] {dest}  ← 插针{int(w*100)}/v2{int((1-w)*100)}组合  总收益 {total*100:+.1f}%  "
          f"回撤 {dd*100:.1f}%  Sharpe {sharpe:.2f}  命中 {hit*100:.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=W_PIN, help="插针权重（0~1）")
    a = ap.parse_args()
    build(a.w)
