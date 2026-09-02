# -*- coding: utf-8 -*-
"""四套方案纸面开仓入口（等价于 python -m paper，保留此文件名便于快速运行）。

用法（在项目根目录下）:
    python paper_four.py            # 结算旧仓 + 换新仓 + 推四套净值
    python paper_four.py --reset    # 清空状态
"""
from __future__ import annotations

import sys

from paper.engine import main

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
