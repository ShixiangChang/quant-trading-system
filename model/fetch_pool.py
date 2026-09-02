# -*- coding: utf-8 -*-
"""拉取自定义池的 K 线 + 资金费率（增量缓存，有缓存自动跳过）。

用法（在项目根目录下）:
    python -m model.fetch_pool                 # 拉 data/model_out/probe_pool.txt 里的池
    python -m model.fetch_pool 我的池.txt      # 指定清单文件名
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from . import config, data


def fetch_pool(pool_file: str = "probe_pool.txt") -> tuple[int, int]:
    path = Path(config.OUTPUT_DIR) / pool_file
    pool = path.read_text(encoding="utf-8").split()
    ok = fail = 0
    for i, sym in enumerate(pool, 1):
        try:
            k = data.fetch_klines(sym)
            data.fetch_funding(sym)
            ok += 1
            print(f"[{i}/{len(pool)}] {sym}: {len(k)} 根K线", flush=True)
        except Exception as exc:  # 个别币失败不阻塞整体，继续拉其它
            fail += 1
            print(f"[{i}/{len(pool)}] {sym} 失败: {exc}", flush=True)
    return ok, fail


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    ok, fail = fetch_pool(sys.argv[1] if len(sys.argv) > 1 else "probe_pool.txt")
    print(f"[fetch_pool] 完成: 成功 {ok} / 失败 {fail}，耗时 {(time.time() - t0) / 60:.1f} 分", flush=True)