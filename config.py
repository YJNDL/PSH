# -*- coding: utf-8 -*-
"""默认配置、CLI 覆盖和派生扫描网格。"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .models import DebugOptions, DerivedScanConfig, FailurePolicyOptions, ScanConfig

# =============================================================================
# Program metadata + editable defaults
# =============================================================================
PROGRAM_NAME = "ECP-LandauScan"
PROGRAM_TAG = "ECP-LS"
PROGRAM_VERSION = "0.9.0-allsg-nostrain"

# PARENT_POSCAR 是默认父相结构文件；命令行 --parent-poscar 只在显式传入时覆盖它。
PARENT_POSCAR = "156.vasp"   # auto-set by prepare_phase_hunter_jobs.py

# OUTPUT_DIR 是扫描结果根目录。CREATE_RUN_SUBDIR=True 时每次运行会在其下新建 run 子目录。
OUTPUT_DIR = "ECP_LandauScan_outputs"

# 结构写出策略：
# - "hits"  : 只写目标空间群 TARGET_SGS 命中的结构
# - "all"   : 写出每个 trial 结构，适合小规模调试，但大扫描会产生大量 POSCAR
# - "per_sg": 每个识别到的空间群最多写 MAX_STRUCTURES_PER_SG 个结构
SAVE_POLICY = "all"  # "hits" | "all" | "per_sg"
MAX_STRUCTURES_PER_SG: Optional[int] = None  # None -> unlimited

CREATE_RUN_SUBDIR = True
RUN_TAG: Optional[str] = None

SYMPREC_PARENT = 0.1
SYMPREC_IDENTIFY = 0.1
ANGLE_TOLERANCE = -1.0

# SCAN_PROFILE 决定振幅、rho、strain 网格密度；派生网格在配置构建阶段集中解析。
SCAN_PROFILE = "dense"  # "fast" | "standard" | "dense"

# TARGET_SGS 为空时不把任何空间群标记为 hit，但仍记录所有 trial 的识别结果。
# 做全空间群探索时通常保持为空，并配合 SAVE_POLICY="per_sg" 或 JSONL 后处理。
TARGET_SGS: List[int] = []

# ENABLE_STRAIN=False 时晶胞固定，派生 strain 网格强制为 sa=sb=sc=1.0。
ENABLE_STRAIN = False

CONFIRMED_COMBOS: List[Dict[str, Any]] = [
    # Example:
    # {"target_sgs": {26}, "mode_indices": [("KPT_R_ARM00_BLOCK001", 0)], "op_directions": [[1.0]]},
]

PARENT_POINT_GROUP_HINT: Optional[str] = None

# STRICT_POINTGROUP_MATCH=True 时，spglib 识别点群与用户 hint 不一致会直接报错；
# 这用于避免在错误父相点群下继续构造模式基。
STRICT_POINTGROUP_MATCH = True

# Optional SLURM helper:
#   python this_script.py --write-slurm
#   python this_script.py --submit-slurm
SLURM_PARTITION: Optional[str] = "h"       # e.g. "cpu" or "compute"
SLURM_ACCOUNT: Optional[str] = None
SLURM_QOS: Optional[str] = None
SLURM_JOB_NAME: Optional[str] = "xyx-tt"   # None -> auto from RUN_TAG / POSCAR / script name
SLURM_NODES = 1
SLURM_NTASKS = 1
SLURM_CPUS_PER_TASK = 128
SLURM_MEM: Optional[str] = None            # e.g. "16G"; set None to omit
SLURM_TIME = "24:00:00"
SLURM_OUTPUT_PATTERN = "%j.out"
SLURM_ERROR_PATTERN = "%j.err"
SLURM_WORKDIR: Optional[str] = None        # None -> use the directory where `sbatch` is called
SLURM_PYTHON_CMD = "python -u"
SLURM_SCRIPT_PATH: Optional[str] = None    # None -> use this script file name in the submit dir
SLURM_EXTRA_SETUP_COMMANDS: List[str] = [
    "source /public/software/intel2018/intel2018_env.sh",
]
SLURM_EXPORT_ENV: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
SLURM_SBATCH_EXTRA_LINES: List[str] = [
    # "#SBATCH --constraint=zen4",
]

ENABLE_RUNTIME_LOG = True
RUNTIME_LOG_FILENAME = "run.log"

# checkpoint/flush 参数：主进程按记录数周期性落盘，便于长任务中断后恢复。
FLUSH_EVERY_N_RECORDS = 10000       # 每多少条记录强制 flush+fsync & 写 checkpoint
PRINT_EVERY_N_TRIALS = 200          # 每多少个 trial 打印一次进度（你可改成 50/100/500 等）
PRINT_ALSO_ON_HIT = True            # 命中目标 SG 时立即打印

# 并行执行参数：
# - N_WORKERS=None 表示按 SLURM_CPUS_PER_TASK 或本机 CPU 数自动推断
# - RESULT_WRITE_IN_MAIN=True 保证 JSONL/CSV/checkpoint/POSCAR 只由主进程写入，顺序可复现
N_WORKERS: Optional[int] = 32
PARALLEL_BATCH_SIZE: Optional[int] = 1024
PARALLEL_MAP_CHUNKSIZE = 16
RESULT_WRITE_IN_MAIN = True
WRITE_RESULTS_CSV = True

AUTO_COMBO_MAX_SIZE = 2
AUTO_DIR_POOL_2D = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, -1.0],
]

RANDOM_OP_DIRECTION_SEED = 12345

# Mode block 数值分解容差；用于 primitive-k commutant/eigen block 聚类。
MODE_BLOCK_TOL = 1e-5

# 结构维度控制 high-symmetry k 点选择：
# - "2d": 只保留 backend primitive reciprocal basis 中 kz=0 的高对称点；
# - "3d": 保留完整三维 BZ 高对称点。
STRUCTURE_DIMENSIONALITY = "2d"

# High-symmetry k-point mode generation:
# fail-fast 主路径固定使用 SeeK-path。不要自动切换其他 k-path backend，
# 也不要输出 complex-only 模式；所有 high-k 模式必须实化到 commensurate supercell。
REQUIRED_KPATH_BACKEND = "seekpath"
HIGH_SYMMETRY_KPOINT_SELECTION = "path_endpoints"  # "path_endpoints" | "all_point_coords" | "labels"
HIGH_SYMMETRY_KPOINT_LABELS: Optional[List[str]] = None
INCLUDE_GAMMA_HIGH_SYMMETRY = True
EXTRA_TARGET_KPOINTS_FRACTIONAL: Optional[List[List[float]]] = None
EXTRA_TARGET_KPOINT_LABELS: Optional[List[str]] = None
EXTRA_TARGET_KPOINTS_BASIS = "primitive"
KPOINT_RATIONALIZE_MAX_DEN = 24
KPOINT_TOL = 1e-8
HIGH_SYMMETRY_MAX_SUPERCELL_SIZE = 512
REAL_MODE_STRATEGY = "cos_sin"

# 统一失败策略：
# - strict: 默认 fail-fast，关键路径失败直接报错；
# - debug: 允许部分调试阶段继续，但必须显式 warning；
# - permissive: 批量探索用，允许跳过失败候选并输出统计/原因。
FAILURE_POLICY = "strict"
FAILURE_POLICY_VALUES = {"strict", "debug", "permissive"}

# Distorted output reduction:
# 默认开启：只在 trial 位移全部施加完、写 POSCAR 前，尝试寻找畸变结构自身的
# primitive cell；不会强制回到父相 primitive cell。极小位移默认跳过，避免
# spglib 把数值噪声误判回高对称父相。
REDUCE_DISTORTED_OUTPUT_CELL = True
REQUIRED_REDUCTION_BACKEND = "spglib"
REDUCE_DISTORTED_SYMPREC = 1e-5
REDUCE_DISTORTED_ANGLE_TOLERANCE = -1.0
REDUCE_DISTORTED_MIN_VOLUME_RATIO = 1.05
REDUCE_DISTORTED_VERIFY = True
REDUCE_DISTORTED_VERIFY_TOL = 1e-5
REDUCE_DISTORTED_KEEP_UNREDUCED_COPY = True
REDUCE_DISTORTED_STRICT = False
REDUCE_DISTORTED_SKIP_IF_AMPLITUDE_BELOW = 1e-4
# None 表示所有写出的结构都可尝试；设为 N 时，只对 atom 数大于 N 的结构尝试 reduction。
REDUCE_DISTORTED_MAX_ATOMS: Optional[int] = None
REDUCE_DISTORTED_LOG_MAPPING = True

_SCAN_PROFILES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "AMP_GRID": [-0.6, 0.6],
        "RHO_GRID": [0.6, 0.9, 1.2],
        "STRAIN_A_GRID": [1.0],
        "STRAIN_B_GRID": [1.0],
        "STRAIN_C_GRID": [1.0],
        "N_RANDOM_DIRS": 0,
    },
    "standard": {
        "AMP_GRID": [-0.8, -0.4, 0.4, 0.8],
        "RHO_GRID": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        "STRAIN_A_GRID": [float(x) for x in np.round(np.arange(0.80, 0.91, 0.02), 4)],
        "STRAIN_B_GRID": [float(x) for x in np.round(np.arange(0.88, 0.97, 0.02), 4)],
        "STRAIN_C_GRID": [float(x) for x in np.round(np.arange(1.00, 1.04, 0.01), 4)],
        "N_RANDOM_DIRS": 30,
    },
    "dense": {
        "AMP_GRID": [-1.0, -0.8, -0.6, -0.4, 0.4, 0.6, 0.8, 1.0],
        "RHO_GRID": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2],
        "STRAIN_A_GRID": [float(x) for x in np.round(np.arange(0.78, 0.93, 0.01), 4)],
        "STRAIN_B_GRID": [float(x) for x in np.round(np.arange(0.86, 0.99, 0.01), 4)],
        "STRAIN_C_GRID": [float(x) for x in np.round(np.arange(0.98, 1.06, 0.01), 4)],
        "N_RANDOM_DIRS": 60,
    },
}

DEBUG_STAGE_NAMES = [
    "config",
    "paths",
    "parent",
    "symmetry",
    "basis",
    "plan",
    "single",
    "combo",
    "summary",
]

RECORD_SCHEMA = [
    "trial",
    "type",
    "mode_indices",
    "op_direction",
    "amps",
    "sa", "sb", "sc",
    "sg", "sg_sym", "pg",
    "hit",
    "label",
    "poscar",
    "combo_target_sgs",
]

def build_failure_policy_options(policy: str) -> FailurePolicyOptions:
    """把单一 failure policy 解析成主流程可读的行为选项。"""
    normalized = str(policy).strip().lower()
    if normalized not in FAILURE_POLICY_VALUES:
        raise ValueError(
            "FAILURE_POLICY must be one of "
            f"{sorted(FAILURE_POLICY_VALUES)}; got {policy!r}."
        )
    if normalized == "strict":
        return FailurePolicyOptions(
            name="strict",
            empty_combo_scan_continues=False,
            empty_high_k_modes_continue=False,
            failed_candidate_continues=False,
        )
    if normalized == "debug":
        return FailurePolicyOptions(
            name="debug",
            empty_combo_scan_continues=True,
            empty_high_k_modes_continue=False,
            failed_candidate_continues=True,
        )
    return FailurePolicyOptions(
        name="permissive",
        empty_combo_scan_continues=True,
        empty_high_k_modes_continue=True,
        failed_candidate_continues=True,
    )


def failure_policy_warning(config: ScanConfig, message: str) -> None:
    """在非 strict 策略允许继续时，统一打印可 grep 的 warning。"""
    print(f"[FAILURE-POLICY][WARN] policy={config.failure_policy}: {message}")


CLI_OVERRIDES: Dict[str, Tuple[str, Callable[[Any], Any]]] = {
    "parent_poscar": ("parent_poscar", str),
    "output_dir": ("output_dir", str),
    "structure_dimensionality": ("structure_dimensionality", lambda value: str(value).lower()),
    "high_symmetry_kpoint_selection": ("high_symmetry_kpoint_selection", str),
    "reduce_distorted_symprec": ("reduce_distorted_symprec", float),
    "reduce_distorted_skip_if_amplitude_below": ("reduce_distorted_skip_if_amplitude_below", float),
}


def build_default_scan_config() -> ScanConfig:
    values: Dict[str, Any] = {}
    missing_defaults: List[str] = []
    for item in fields(ScanConfig):
        const_name = item.name.upper()
        if const_name not in globals():
            missing_defaults.append(const_name)
            continue
        values[item.name] = copy.deepcopy(globals()[const_name])
    if missing_defaults:
        raise RuntimeError(
            "Missing default constants for ScanConfig fields: "
            + ", ".join(sorted(missing_defaults))
        )
    return ScanConfig(**values)


def build_debug_options_from_args(args: Optional[argparse.Namespace]) -> DebugOptions:
    if args is None:
        return DebugOptions(
            enabled=False,
            stop_after_stage=None,
            no_parallel=False,
            max_trials=None,
            print_config=False,
            print_plan=False,
        )

    stop_after_stage = getattr(args, "debug_stop_after_stage", None)
    max_trials = getattr(args, "debug_max_trials", None)
    if max_trials is not None and int(max_trials) < 0:
        raise ValueError("--debug-max-trials must be >= 0.")
    enabled = bool(
        getattr(args, "debug", False)
        or stop_after_stage is not None
        or getattr(args, "debug_no_parallel", False)
        or max_trials is not None
        or getattr(args, "debug_print_config", False)
        or getattr(args, "debug_print_plan", False)
    )
    return DebugOptions(
        enabled=enabled,
        stop_after_stage=stop_after_stage,
        no_parallel=bool(getattr(args, "debug_no_parallel", False)),
        max_trials=int(max_trials) if max_trials is not None else None,
        print_config=bool(getattr(args, "debug_print_config", False)),
        print_plan=bool(getattr(args, "debug_print_plan", False)),
    )


def build_derived_scan_config(config: ScanConfig) -> DerivedScanConfig:
    if config.scan_profile not in _SCAN_PROFILES:
        raise ValueError(f"Unknown SCAN_PROFILE='{config.scan_profile}'. Choose from {sorted(_SCAN_PROFILES.keys())}")

    profile = _SCAN_PROFILES[config.scan_profile]
    amp_grid = [float(x) for x in profile["AMP_GRID"]]
    rho_grid = [float(x) for x in profile["RHO_GRID"]]
    strain_a_grid = [float(x) for x in profile["STRAIN_A_GRID"]]
    strain_b_grid = [float(x) for x in profile["STRAIN_B_GRID"]]
    strain_c_grid = [float(x) for x in profile["STRAIN_C_GRID"]]
    n_random_dirs = int(profile["N_RANDOM_DIRS"])

    if not bool(config.enable_strain):
        strain_a_grid = [1.0]
        strain_b_grid = [1.0]
        strain_c_grid = [1.0]

    save_policy = str(config.save_policy).strip().lower()
    if save_policy not in {"hits", "all", "per_sg"}:
        raise ValueError(f"Unknown SAVE_POLICY='{config.save_policy}'. Use 'hits'|'all'|'per_sg'.")

    write_all_trial_structures = save_policy == "all"
    write_hit_structures = save_policy == "hits"
    write_per_sg_structures = save_policy == "per_sg"
    target_sg_set = set(int(x) for x in config.target_sgs)

    config_warnings: List[str] = []
    if write_hit_structures and (not target_sg_set):
        config_warnings.append(
            "[WARN] SAVE_POLICY='hits' but TARGET_SGS is empty -> no structures will be written. "
            "Consider SAVE_POLICY='per_sg' (recommended) or 'all'."
        )

    random_op_direction_candidates = max(500, 50 * max(1, n_random_dirs))

    return DerivedScanConfig(
        amp_grid=amp_grid,
        rho_grid=rho_grid,
        strain_a_grid=strain_a_grid,
        strain_b_grid=strain_b_grid,
        strain_c_grid=strain_c_grid,
        enable_random_op_directions=(n_random_dirs > 0),
        n_random_op_directions_per_combo=n_random_dirs,
        random_op_direction_candidates=int(random_op_direction_candidates),
        write_all_trial_structures=bool(write_all_trial_structures),
        write_hit_structures=bool(write_hit_structures),
        write_per_sg_structures=bool(write_per_sg_structures),
        write_limit_per_sg=config.max_structures_per_sg,
        target_sg_set=target_sg_set,
        failure_options=build_failure_policy_options(config.failure_policy),
        config_warnings=config_warnings,
    )


def apply_cli_overrides(config: ScanConfig, args: Optional[argparse.Namespace]) -> ScanConfig:
    """应用 CLI 覆盖；普通字段走映射表，少数特殊参数单独处理。"""
    if args is None:
        return config
    updates: Dict[str, Any] = {}
    for arg_name, (field_name, converter) in CLI_OVERRIDES.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field_name] = converter(value)
    if getattr(args, "reduce_distorted_output_cell", False):
        updates["reduce_distorted_output_cell"] = True
    if getattr(args, "high_symmetry_kpoint_labels", None):
        updates["high_symmetry_kpoint_labels"] = [
            item.strip()
            for item in str(getattr(args, "high_symmetry_kpoint_labels")).split(",")
            if item.strip()
        ]
    if getattr(args, "failure_policy", None):
        updates["failure_policy"] = build_failure_policy_options(str(getattr(args, "failure_policy"))).name
    if getattr(args, "exclude_gamma_high_symmetry", False):
        updates["include_gamma_high_symmetry"] = False
    return replace(config, **updates) if updates else config


def validate_scan_config(config: ScanConfig) -> None:
    """集中校验配置，不在 import 阶段执行。"""
    if str(config.structure_dimensionality).lower() not in {"2d", "3d"}:
        raise ValueError("STRUCTURE_DIMENSIONALITY must be '2d' or '3d'.")
    if str(config.high_symmetry_kpoint_selection).strip().lower() not in {"path_endpoints", "all_point_coords", "labels"}:
        raise ValueError("HIGH_SYMMETRY_KPOINT_SELECTION must be 'path_endpoints', 'all_point_coords', or 'labels'.")
    build_failure_policy_options(config.failure_policy)
    if float(config.reduce_distorted_min_volume_ratio) < 1.0:
        raise ValueError("REDUCE_DISTORTED_MIN_VOLUME_RATIO must be >= 1.0.")


def build_scan_config_from_defaults_and_cli(args: Optional[argparse.Namespace]) -> Tuple[ScanConfig, DerivedScanConfig, DebugOptions]:
    """构建最终配置：defaults -> CLI overrides -> validate -> derived/debug。"""
    config = build_default_scan_config()
    config = apply_cli_overrides(config, args)
    validate_scan_config(config)
    derived = build_derived_scan_config(config)
    debug = build_debug_options_from_args(args)
    return config, derived, debug


def check_required_dependencies() -> None:
    """进入主扫描前检查 fail-fast 主路径依赖。"""
    missing: List[str] = []
    for module_name, install_hint in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("spglib", "spglib"),
        ("seekpath", "seekpath"),
    ]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(install_hint)
    if missing:
        raise RuntimeError(
            "Missing required dependencies for the fail-fast main workflow: "
            f"{', '.join(missing)}. Install them in the active Python environment."
        )
