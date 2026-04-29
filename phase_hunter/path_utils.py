# -*- coding: utf-8 -*-
"""路径和文件名小工具。

本模块只依赖标准库，供 SLURM、runtime path 和结构文件命名复用，避免这些模块
为了一个文件名 helper 反向依赖 POSCAR IO。
"""
from __future__ import annotations

import re


def _safe_filename(s: str, max_len: int = 180) -> str:
    """生成适合作为文件名片段的字符串。"""
    s = str(s).strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = s.replace(" ", "_")
    s = re.sub(r"__+", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s.strip("._")

