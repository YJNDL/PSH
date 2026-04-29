# -*- coding: utf-8 -*-
"""debug 输出和阶段停止工具。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .models import DebugOptions, DerivedScanConfig, LandauBasisData, RunSummary, ScanConfig, ScanPlan

def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_json_safe(x) for x in value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def debug_stage_event(stage: str, event: str, debug: DebugOptions, details: Optional[Dict[str, Any]] = None) -> None:
    if not debug.enabled:
        return
    print()
    print(f"================= DEBUG {event.upper()} {stage} =================")
    if details:
        for key, value in details.items():
            print(f"[DEBUG] {key}: {_json_safe(value)}")
    print("====================================================")


def print_effective_config(config: ScanConfig, derived: DerivedScanConfig, debug: DebugOptions) -> None:
    payload = {
        "config": _json_safe(asdict(config)),
        "derived": _json_safe(asdict(derived)),
        "debug": _json_safe(asdict(debug)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def should_stop_after_stage(debug: DebugOptions, stage: str) -> bool:
    if debug.stop_after_stage != stage:
        return False
    if debug.enabled:
        print(f"[DEBUG] stop requested after stage '{stage}'.")
    return True

def print_scan_plan_summary(plan: ScanPlan, landau_data: LandauBasisData, derived: DerivedScanConfig) -> None:
    print()
    print("================= Debug Scan Plan =================")
    print(f"[DEBUG] mode blocks = {len(landau_data.mode_keys)}")
    print(f"[DEBUG] total modes = {sum(len(v) for v in landau_data.landau_basis.values())}")
    print(f"[DEBUG] single strains = {len(plan.single_phase.single_strains)}")
    print(f"[DEBUG] single amps = {len(plan.single_phase.amp_grid)}")
    print(f"[DEBUG] single total tasks = {plan.single_phase.total_tasks}")
    print(f"[DEBUG] combo specs = {len(plan.combo_phase.specs)}")
    print(f"[DEBUG] combo rhos = {len(plan.combo_phase.rho_grid)}")
    print(f"[DEBUG] combo strains = {len(plan.combo_phase.combo_strains)}")
    print(f"[DEBUG] combo total tasks = {plan.combo_phase.total_tasks}")
    print(f"[DEBUG] random combo dirs enabled = {derived.enable_random_op_directions}")
    print("===================================================")


def print_run_summary(summary: RunSummary) -> None:
    """打印最终运行摘要；只读 RunSummary，不修改运行态。"""
    uniq = sorted(summary.sg_counter.keys())
    print()
    print("================= 扫描结束 Summary =================")
    print(f"[INFO] total records = {summary.total_records}")
    print(f"[INFO] unique space-groups found = {uniq}")
    print("===================================================")
    print(f"[INFO] results saved to JSONL: {summary.jsonl}")
    if summary.csv:
        print(f"[INFO] results saved to CSV : {summary.csv}")
    if summary.checkpoint:
        print(f"[INFO] checkpoint saved to    : {summary.checkpoint}")
