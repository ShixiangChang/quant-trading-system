# -*- coding: utf-8 -*-
"""数据归档：把 model_cache 自拉数据（1h K 线 / 资金费率 / 1h 微观结构）导入 monitor.db，统一数据层。

这是「接通断裂」的第一步：model 自拉的历史数据不再散落在 CSV 里，而是与 monitor
的细粒度数据落在同一个 SQLite 库，未来 feature 层从一处取数。

用法（在项目根目录下）：
    python -m model.archive

幂等：主键 (symbol, open_time/ts/kind) 去重，重复跑不会产生重复行，可放心增量重跑。
"""
from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

from monitor import config as mcfg
from monitor.db import SCHEMA

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "model_cache"
_FNAME = re.compile(r"^(.*)_(\d+)d_(.+)\.csv$")
BATCH = 2000  # SQLite 单次 executemany 变量数上限内的安全批量


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(mcfg.DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    c.executescript(SCHEMA)  # 复用 monitor 的 schema（幂等建表），保证 DDL 单一来源
    return c


def _read(p: Path) -> list[list[str]]:
    with open(p, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # 跳过表头
        return list(reader)


def _f(x: str):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _execmany(conn: sqlite3.Connection, sql: str, data: list, batch: int = BATCH) -> None:
    for i in range(0, len(data), batch):
        conn.executemany(sql, data[i:i + batch])


def archive(cache_dir: Path = CACHE_DIR) -> dict:
    conn = _conn()
    counts = {"klines": 0, "funding_hist": 0, "micro_1h": 0}
    files = sorted(cache_dir.glob("*.csv"))
    for i, p in enumerate(files, 1):
        m = _FNAME.match(p.name)
        if not m:
            continue
        prefix, _days, symbol = m.group(1), m.group(2), m.group(3)
        rows = _read(p)
        if not rows:
            continue

        if prefix == "klines":
            # 列: open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote
            data = [(symbol, int(float(r[0])), *[_f(x) for x in r[1:]]) for r in rows]
            _execmany(conn,
                "INSERT OR REPLACE INTO klines (symbol, open_time, open, high, low, close,"
                " volume, quote_volume, trades, taker_buy_base, taker_buy_quote)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", data)
            counts["klines"] += len(data)
        elif prefix == "funding":
            # 列: funding_time, funding
            data = [(symbol, int(float(r[0])), _f(r[1])) for r in rows]
            _execmany(conn,
                "INSERT OR REPLACE INTO funding_hist (symbol, funding_time, funding)"
                " VALUES (?,?,?)", data)
            counts["funding_hist"] += len(data)
        elif prefix.startswith("micro_"):
            # 列: ts, v；kind = oi / lsr / top_lsr / taker_bs
            kind = prefix[len("micro_"):]
            data = [(symbol, int(float(r[0])), kind, _f(r[1])) for r in rows]
            _execmany(conn,
                "INSERT OR REPLACE INTO micro_1h (symbol, ts, kind, value)"
                " VALUES (?,?,?,?)", data)
            counts["micro_1h"] += len(data)

        conn.commit()
        if i % 100 == 0:
            print(f"[archive] 进度 {i}/{len(files)} ...")
    conn.close()
    return counts


if __name__ == "__main__":
    c = archive()
    print(f"[archive] 完成: klines={c['klines']:,}  funding_hist={c['funding_hist']:,}  "
          f"micro_1h={c['micro_1h']:,}")
