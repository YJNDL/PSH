# -*- coding: utf-8 -*-
"""SLURM 脚本生成和提交工具。"""
from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

from .config import PROGRAM_TAG, build_default_scan_config
from .models import ScanConfig
from .path_utils import _safe_filename

def _shell_quote(value: Union[str, Path]) -> str:
    return shlex.quote(str(value).replace("\\", "/"))


def _default_slurm_job_name(config: ScanConfig) -> str:
    for candidate in [config.run_tag, Path(config.parent_poscar).stem if config.parent_poscar else None, Path("run_phase_hunter.py").stem]:
        safe = _safe_filename(candidate or "", max_len=80)
        if safe:
            return safe
    return PROGRAM_TAG.lower()


def _default_slurm_submit_path(config: ScanConfig) -> Path:
    stem = _safe_filename(Path("run_phase_hunter.py").stem, max_len=80) or PROGRAM_TAG.lower()
    if config.run_tag:
        stem = _safe_filename(f"{stem}_{config.run_tag}", max_len=80)
    return Path(f"submit_{stem}.slurm")


def build_slurm_submit_script_text(config: Optional[ScanConfig] = None) -> str:
    config = config or build_default_scan_config()
    script_target = str(config.slurm_script_path).strip() if config.slurm_script_path else "run_phase_hunter.py"
    job_name_raw = str(config.slurm_job_name).strip() if config.slurm_job_name else _default_slurm_job_name(config)
    job_name = _safe_filename(job_name_raw, max_len=80) or PROGRAM_TAG.lower()
    python_cmd = str(config.slurm_python_cmd).strip() or "python3"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={int(config.slurm_nodes)}",
        f"#SBATCH --ntasks={int(config.slurm_ntasks)}",
        f"#SBATCH --cpus-per-task={int(config.slurm_cpus_per_task)}",
        f"#SBATCH --time={str(config.slurm_time).strip()}",
        f"#SBATCH --output={str(config.slurm_output_pattern).strip()}",
        f"#SBATCH --error={str(config.slurm_error_pattern).strip()}",
    ]

    for flag, value in [
        ("--partition", config.slurm_partition),
        ("--account", config.slurm_account),
        ("--qos", config.slurm_qos),
        ("--mem", config.slurm_mem),
    ]:
        if value is not None and str(value).strip():
            lines.append(f"#SBATCH {flag}={str(value).strip()}")

    for extra_line in config.slurm_sbatch_extra_lines:
        line = str(extra_line).strip()
        if line:
            lines.append(line if line.startswith("#SBATCH") else f"#SBATCH {line}")

    lines.extend(["", "set -euo pipefail", ""])

    if config.slurm_workdir and str(config.slurm_workdir).strip():
        lines.append(f"cd {_shell_quote(str(config.slurm_workdir).strip())}")
    else:
        lines.append('cd "${SLURM_SUBMIT_DIR:-$PWD}"')

    for cmd in config.slurm_extra_setup_commands:
        cmd_line = str(cmd).rstrip()
        if cmd_line:
            lines.append(cmd_line)

    if config.slurm_export_env:
        for env_key, env_value in config.slurm_export_env.items():
            env_name = str(env_key).strip()
            if env_name:
                lines.append(f"export {env_name}={shlex.quote(str(env_value))}")

    lines.extend(["", f"{python_cmd} {_shell_quote(script_target)} \"$@\"", ""])
    return "\n".join(lines)


def write_slurm_submit_script(path: Optional[Union[str, Path]] = None, config: Optional[ScanConfig] = None) -> Path:
    config = config or build_default_scan_config()
    out_path = Path(path) if path else _default_slurm_submit_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_slurm_submit_script_text(config=config)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    try:
        out_path.chmod(0o755)
    except OSError:
        pass
    return out_path.resolve()


def submit_slurm_script(path: Union[str, Path]) -> str:
    sbatch_path = shutil.which("sbatch")
    if sbatch_path is None:
        raise RuntimeError("`sbatch` not found in PATH. Please run `--submit-slurm` on a SLURM login node, or use `--write-slurm` first.")

    result = subprocess.run(
        [sbatch_path, str(Path(path))],
        check=True,
        capture_output=True,
        text=True,
    )
    message = result.stdout.strip() or result.stderr.strip() or f"Submitted {path}"
    print(f"[SLURM] {message}")
    return message
