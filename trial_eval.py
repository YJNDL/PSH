# -*- coding: utf-8 -*-
"""trial 计算和 worker 入口。

worker 只负责把 TrialTask 计算成 TrialEvaluation，不写 JSONL/CSV/POSCAR/checkpoint；
所有持久化副作用由主进程 scan_engine.run_trial_main_loop() 完成。
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import normalize_direction
from .models import (
    ComboTrialTask,
    DerivedScanConfig,
    LandauBasisData,
    ParentStructureData,
    ScanConfig,
    SingleTrialTask,
    TrialEvaluation,
    TrialEvaluationContext,
    TrialTask,
)
from .structure_ops import make_distorted_crystal_with_displacement
from .symmetry import identify_spacegroup, normalize_pointgroup_symbol

def _source_cell_metadata(
    mode_amplitudes: Dict[Tuple[str, int], float],
    trial_context: TrialEvaluationContext,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """提取本 trial 使用的 mode cell 信息，供输出 reduction metadata 记录来源。"""
    cell_kinds = []
    matrices = []
    for mode_key, _mode_idx in mode_amplitudes:
        actual_key = mode_key
        if actual_key not in trial_context.landau_basis and actual_key in trial_context.mode_key_aliases:
            actual_key = trial_context.mode_key_aliases[actual_key]
        cell = trial_context.mode_cell_by_key.get(actual_key)
        if cell is None:
            continue
        cell_kinds.append(str(cell.cell_kind))
        if cell.supercell_matrix is not None:
            matrices.append(np.asarray(cell.supercell_matrix, dtype=int))
    source_kind = sorted(set(cell_kinds))[0] if cell_kinds else "parent"
    source_matrix = matrices[0].copy() if matrices else None
    return source_kind, source_matrix


def _single_mode_amplitudes(task: SingleTrialTask) -> Dict[Tuple[str, int], float]:
    return {(task.mode_key, int(task.mode_idx)): float(task.amp)}


def _combo_mode_amplitudes(task: ComboTrialTask) -> Tuple[Dict[Tuple[str, int], float], List[float]]:
    d = normalize_direction([float(x) for x in task.op_direction])
    mode_amplitudes = {
        (task.mode_indices[k][0], int(task.mode_indices[k][1])): float(task.rho) * float(d[k])
        for k in range(len(task.mode_indices))
    }
    return mode_amplitudes, [float(x) for x in d]


def evaluate_trial_task(task: TrialTask, trial_context: TrialEvaluationContext) -> TrialEvaluation:
    if isinstance(task, SingleTrialTask):
        mode_amplitudes = _single_mode_amplitudes(task)
        source_kind, source_matrix = _source_cell_metadata(mode_amplitudes, trial_context)
        trial, displacement = make_distorted_crystal_with_displacement(
            trial_context.parent,
            trial_context.landau_basis,
            mode_amplitudes,
            task.sa,
            task.sb,
            task.sc,
            trial_context.mode_key_aliases,
            trial_context.mode_cell_by_key,
        )
        sg, sg_sym, pg_trial = identify_spacegroup(
            trial,
            symprec=trial_context.symprec_identify,
            angle_tolerance=trial_context.angle_tolerance,
        )
        hit = (int(sg) in trial_context.target_sg_set) if trial_context.target_sg_set else False
        label = f"{task.mode_key}[{task.mode_idx}]_amp={task.amp:+.3f}_sa={task.sa:.3f}_sb={task.sb:.3f}_sc={task.sc:.3f}"
        return TrialEvaluation(
            kind="single",
            label=label,
            trial_crystal=trial,
            sg=int(sg),
            sg_sym=str(sg_sym),
            pg=normalize_pointgroup_symbol(pg_trial),
            hit=bool(hit),
            mode_indices=[(task.mode_key, int(task.mode_idx))],
            op_direction=None,
            amps=[float(task.amp)],
            sa=float(task.sa),
            sb=float(task.sb),
            sc=float(task.sc),
            combo_target_sgs=None,
            displacement=displacement,
            source_mode_cell_kind=source_kind,
            source_supercell_matrix=source_matrix,
        )

    mode_amplitudes, d = _combo_mode_amplitudes(task)
    source_kind, source_matrix = _source_cell_metadata(mode_amplitudes, trial_context)
    trial, displacement = make_distorted_crystal_with_displacement(
        trial_context.parent,
        trial_context.landau_basis,
        mode_amplitudes,
        task.sa,
        task.sb,
        task.sc,
        trial_context.mode_key_aliases,
        trial_context.mode_cell_by_key,
    )
    sg, sg_sym, pg_trial = identify_spacegroup(
        trial,
        symprec=trial_context.symprec_identify,
        angle_tolerance=trial_context.angle_tolerance,
    )
    hit = (int(sg) in trial_context.target_sg_set) if trial_context.target_sg_set else False
    label = f"{task.mode_indices}_dir={d}_rho={task.rho:.3f}_sa={task.sa:.3f}_sb={task.sb:.3f}_sc={task.sc:.3f}"
    return TrialEvaluation(
        kind="combo",
        label=label,
        trial_crystal=trial,
        sg=int(sg),
        sg_sym=str(sg_sym),
        pg=normalize_pointgroup_symbol(pg_trial),
        hit=bool(hit),
        mode_indices=[(m[0], int(m[1])) for m in task.mode_indices],
        op_direction=[float(x) for x in d],
        amps=[float(mode_amplitudes[(m[0], int(m[1]))]) for m in task.mode_indices],
        sa=float(task.sa),
        sb=float(task.sb),
        sc=float(task.sc),
        combo_target_sgs=copy.deepcopy(task.combo_target_sgs),
        displacement=displacement,
        source_mode_cell_kind=source_kind,
        source_supercell_matrix=source_matrix,
    )


_WORKER_CONTEXT: Optional[TrialEvaluationContext] = None


def _init_parallel_worker(trial_context: TrialEvaluationContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = trial_context


def _evaluate_trial_task_from_worker(task: TrialTask) -> TrialEvaluation:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Parallel worker was not initialized.")
    return evaluate_trial_task(task, _WORKER_CONTEXT)

def build_trial_evaluation_context(
    parent_data: ParentStructureData,
    landau_data: LandauBasisData,
    config: ScanConfig,
    derived: DerivedScanConfig,
) -> TrialEvaluationContext:
    return TrialEvaluationContext(
        parent=parent_data.crystal,
        landau_basis=landau_data.landau_basis,
        symprec_identify=float(config.symprec_identify),
        angle_tolerance=float(config.angle_tolerance),
        target_sg_set=set(derived.target_sg_set),
        mode_key_aliases=dict(landau_data.mode_key_aliases),
        mode_cell_by_key=dict(landau_data.mode_cell_by_key),
    )
