# -*- coding: utf-8 -*-
"""命令行入口。

cli.py 只负责 argparse 和 SLURM 分支；本地扫描主流程显式写在 run_phase_hunter.py。
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from .config import DEBUG_STAGE_NAMES, PROGRAM_NAME, PROGRAM_TAG, build_scan_config_from_defaults_and_cli
from .slurm import _shell_quote, submit_slurm_script, write_slurm_submit_script

def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析 CLI。

    普通运行保持历史默认；debug 参数只影响本次进程，适合本地单步调试和小样本 smoke test。
    """
    parser = argparse.ArgumentParser(
        description=f"{PROGRAM_NAME} ({PROGRAM_TAG})",
        epilog=(
            "Examples:\n"
            "  python this_script.py\n"
            "  python this_script.py --write-slurm\n"
            "  python this_script.py --submit-slurm\n"
            "  python this_script.py --parent-poscar tests/fixtures/POSCAR_cubic --debug --debug-print-plan --debug-stop-after-stage plan\n"
            "  python this_script.py --debug --debug-print-config --debug-stop-after-stage config\n"
            "  python this_script.py --debug --debug-no-parallel --debug-max-trials 2"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--write-slurm",
        action="store_true",
        help="Generate an sbatch helper script and exit.",
    )
    parser.add_argument(
        "--submit-slurm",
        action="store_true",
        help="Generate the sbatch helper script, submit it with sbatch, and exit.",
    )
    parser.add_argument(
        "--slurm-file",
        default=None,
        help="Output path for the generated sbatch script.",
    )
    # 调试覆盖参数：只在命令行显式传入时替换默认配置，不改变文件顶部默认值。
    parser.add_argument(
        "--parent-poscar",
        default=None,
        help="Override the default parent POSCAR path for this run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the default output directory for this run.",
    )
    parser.add_argument(
        "--structure-dimensionality",
        choices=["2d", "3d"],
        default=None,
        help="Set structure dimensionality for high-k selection. 2d keeps only kz=0 special points; 3d keeps full BZ.",
    )
    parser.add_argument(
        "--high-symmetry-kpoint-selection",
        choices=["path_endpoints", "all_point_coords", "labels"],
        default=None,
        help="Select which high-symmetry k points to analyze.",
    )
    parser.add_argument(
        "--high-symmetry-kpoint-labels",
        default=None,
        help="Comma-separated k labels used when --high-symmetry-kpoint-selection=labels.",
    )
    parser.add_argument(
        "--failure-policy",
        choices=["strict", "debug", "permissive"],
        default=None,
        help=(
            "Unified failure policy. strict is fail-fast; debug/permissive may continue "
            "only with explicit warnings and skip summaries."
        ),
    )
    parser.add_argument(
        "--exclude-gamma-high-symmetry",
        action="store_true",
        help="Do not auto-add Gamma/G when high-symmetry k-point modes are enabled.",
    )
    parser.add_argument(
        "--reduce-distorted-symprec",
        type=float,
        default=None,
        help="Symmetry tolerance used only for distorted output reduction.",
    )
    parser.add_argument(
        "--reduce-distorted-skip-if-amplitude-below",
        type=float,
        default=None,
        help="Skip output-cell reduction if max displacement norm is below this threshold.",
    )
    # 以下 debug 选项默认关闭，主要用于本地验证阶段边界、任务顺序和 checkpoint 行为。
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra stage banners and debug-oriented diagnostics.",
    )
    parser.add_argument(
        "--debug-stop-after-stage",
        choices=DEBUG_STAGE_NAMES,
        default=None,
        help="Stop safely after the named stage.",
    )
    parser.add_argument(
        "--debug-no-parallel",
        action="store_true",
        help="Force single-process execution for debugging.",
    )
    parser.add_argument(
        "--debug-max-trials",
        type=int,
        default=None,
        help="Process at most N new trials in this invocation.",
    )
    parser.add_argument(
        "--debug-print-config",
        action="store_true",
        help="Print the effective config and derived config.",
    )
    parser.add_argument(
        "--debug-print-plan",
        action="store_true",
        help="Print the single/combo scan plan summary before execution.",
    )
    return parser.parse_args(argv)



def main(argv: Optional[List[str]] = None) -> None:
    """兼容入口：解析 CLI 并处理 SLURM 分支。

    普通本地扫描请直接运行 run_phase_hunter.py。这样主流程留在入口脚本中，
    便于在 PyCharm 里按真实执行顺序阅读和下断点。
    """
    args = parse_cli_args(argv)

    if args.write_slurm or args.submit_slurm:
        _run_slurm_cli_branch(args)
        return

    raise SystemExit("普通本地扫描请运行 `python run_phase_hunter.py`；显式主流程保留在入口脚本中。")

def _run_slurm_cli_branch(args: argparse.Namespace) -> None:
    """处理 SLURM 写脚本/提交分支；不进入扫描主流程。"""
    config, _, _ = build_scan_config_from_defaults_and_cli(args)
    slurm_path = write_slurm_submit_script(args.slurm_file, config=config)
    print(f"[SLURM] submit script written to: {slurm_path}")
    print(f"[SLURM] run manually with: sbatch {_shell_quote(slurm_path)}")
    if args.submit_slurm:
        submit_slurm_script(slurm_path)
