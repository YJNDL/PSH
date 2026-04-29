# -*- coding: utf-8 -*-
"""运行目录、父结构加载和 runtime logging。

POSCAR 解析/写出留在 io_poscar.py；这里集中处理运行时文件系统副作用：
读取父相结构、创建 run_dir、配置 stdout/stderr tee 日志。
"""
from __future__ import annotations

import atexit
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

from .debug_tools import debug_stage_event
from .io_poscar import read_poscar
from .models import DebugOptions, ParentStructureData, ParentSymmetryData, RunPaths, ScanConfig
from .path_utils import _safe_filename


_RUNTIME_LOG_PATH: Optional[Path] = None
_RUNTIME_LOG_FILE: Any = None
_RUNTIME_LOG_CONFIGURED = False


def _enable_stream_line_buffering(stream: Any) -> None:
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


def enable_console_line_buffering() -> None:
    """打开 stdout/stderr 行缓冲，便于长任务实时查看日志。"""
    _enable_stream_line_buffering(sys.stdout)
    _enable_stream_line_buffering(sys.stderr)


def _close_runtime_log_file() -> None:
    """安全关闭运行日志，避免解释器退出时对已关闭文件再次 flush。"""
    global _RUNTIME_LOG_FILE

    log_file = _RUNTIME_LOG_FILE
    if log_file is None:
        return
    try:
        if not log_file.closed:
            log_file.flush()
            log_file.close()
    except Exception:
        pass


class TeeStream:
    """把控制台输出同步写入 runtime log。"""

    def __init__(self, primary: Any, secondary: Any):
        self.primary = primary
        self.secondary = secondary

    @property
    def encoding(self) -> str:
        return getattr(self.primary, "encoding", "utf-8")

    def write(self, data: str) -> int:
        n = self.primary.write(data)
        try:
            self.secondary.write(data)
        except Exception:
            pass
        if "\n" in data:
            self.flush()
        return n

    def flush(self) -> None:
        try:
            self.primary.flush()
        except Exception:
            pass
        try:
            self.secondary.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(self.primary.fileno())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def configure_runtime_logging(run_dir: Union[str, Path], *, enabled: bool, filename: str) -> Optional[Path]:
    """配置 runtime log tee；多次调用时复用已配置日志。"""
    global _RUNTIME_LOG_PATH, _RUNTIME_LOG_FILE, _RUNTIME_LOG_CONFIGURED

    enable_console_line_buffering()
    if not enabled:
        return None
    if _RUNTIME_LOG_CONFIGURED and _RUNTIME_LOG_PATH is not None:
        return _RUNTIME_LOG_PATH

    log_path = Path(run_dir) / str(filename).strip()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    _RUNTIME_LOG_FILE = log_file
    _RUNTIME_LOG_PATH = log_path.resolve()
    _RUNTIME_LOG_CONFIGURED = True
    atexit.register(_close_runtime_log_file)
    return _RUNTIME_LOG_PATH


def load_parent_structure(config: ScanConfig, debug: DebugOptions) -> ParentStructureData:
    """读取父相 POSCAR，输出 ParentStructureData。"""
    debug_stage_event("parent", "begin", debug, {"poscar": config.parent_poscar})
    poscar_path = Path(config.parent_poscar)
    if not poscar_path.is_file():
        raise FileNotFoundError(f"Parent POSCAR not found: {config.parent_poscar}")
    parent = read_poscar(poscar_path)
    print(f"[INFO] 原始父相: {config.parent_poscar}, N = {parent.nsites}")
    parent_data = ParentStructureData(poscar_path=poscar_path, crystal=parent)
    debug_stage_event("parent", "end", debug, {
        "poscar_path": str(poscar_path.resolve()),
        "nsites": parent.nsites,
    })
    return parent_data


def prepare_run_paths_and_logging(config: ScanConfig, symmetry: ParentSymmetryData, debug: DebugOptions) -> RunPaths:
    """准备输出目录和运行日志。

    输出 RunPaths，后续 writer、checkpoint、POSCAR 都写入 run_dir。
    """
    debug_stage_event("paths", "begin", debug, {
        "output_dir": config.output_dir,
        "create_run_subdir": config.create_run_subdir,
    })

    base_out = Path(config.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    run_dir = base_out
    if config.create_run_subdir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parent_stem = Path(config.parent_poscar).stem
        pg_tag = _safe_filename(str(symmetry.point_group_used))
        parts: List[str] = []
        if config.run_tag:
            parts.append(_safe_filename(config.run_tag))
        parts.append(_safe_filename(f"{parent_stem}_SG{int(symmetry.spacegroup_number):03d}_PG{pg_tag}_{ts}"))
        run_dir = base_out / "_".join([p for p in parts if p])
        run_dir.mkdir(parents=True, exist_ok=True)

    log_path = configure_runtime_logging(
        run_dir,
        enabled=bool(config.enable_runtime_log),
        filename=str(config.runtime_log_filename),
    )
    print(f"[INFO] 输出目录: {run_dir.resolve()}")
    if log_path is not None:
        print(f"[INFO] 运行日志: {log_path}")

    paths = RunPaths(output_root=base_out.resolve(), run_dir=run_dir.resolve(), log_path=log_path)
    debug_stage_event("paths", "end", debug, {
        "run_dir": str(paths.run_dir),
        "log_path": str(paths.log_path) if paths.log_path else None,
    })
    return paths

