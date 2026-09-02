# -*- coding: utf-8 -*-
"""把插针抄底策略结果落成「独立轨道」JSON，供看板 status.py 的 renderTracks 展示。

轨道格式（data/model_out/tracks/*.json）：
  name / start / end / stats{total_ret, max_drawdown, hit_rate} / equity[[ts秒, nav], ...]

从 pin_strategy 落盘的 [day编号, nav] 转回 [ts秒, nav]（day*86400）。
用法：python -m backtest.pin_track
"""
from __future__ import annotations

import json
from pathlib import Path

TRACKS_DIR = Path("data/model_out/tracks")

SOURCES = [
    ("pin_12h", "data/model_out/pin_strategy_12h.json", "插针抄底·12h", "★"),
    ("pin_4h", "data/model_out/pin_strategy.json", "插针抄底·4h", ""),
]


def _fmt_day(d: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(d * 86400, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def build() -> None:
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    for stem, src, name, mark in SOURCES:
        p = Path(src)
        if not p.exists():
            print(f"[pin_track] 跳过（缺 {src}）")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        eq_day = d.get("equity", [])
        eq = [[int(day) * 86400, float(nav)] for day, nav in eq_day]
        stats = {
            "total_ret": d.get("total_ret", 0.0),
            "max_drawdown": d.get("max_dd", 0.0),
            "hit_rate": d.get("hit_rate", 0.0),
            "annual": d.get("annual", 0.0),
            "sharpe": d.get("sharpe", 0.0),
            "events": d.get("events", 0),
            "odds": d.get("odds", 0.0),
        }
        out = {
            "name": f"{mark}{name}",
            "start": _fmt_day(eq_day[0][0]) if eq_day else "",
            "end": _fmt_day(eq_day[-1][0]) if eq_day else "",
            "stats": stats,
            "equity": eq,
            "monthly": {},
        }
        dest = TRACKS_DIR / f"{stem}.json"
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[pin_track] {dest}  ← {name}  总收益 {stats['total_ret']*100:+.1f}%  "
              f"回撤 {stats['max_drawdown']*100:.1f}%  Sharpe {stats['sharpe']:.2f}  "
              f"命中 {stats['hit_rate']*100:.0f}%")


if __name__ == "__main__":
    build()
