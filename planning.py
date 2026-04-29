# -*- coding: utf-8 -*-
"""OP 方向、combo 计划和 trial task 生成。"""
from __future__ import annotations

import copy
from itertools import combinations, product
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .config import failure_policy_warning
from .debug_tools import debug_stage_event
from .geometry import normalize_direction as _normalize_direction
from .models import (
    ComboPhasePlan,
    ComboPlanSpec,
    ComboTrialTask,
    DebugOptions,
    DerivedScanConfig,
    LandauBasisData,
    ParentStructureData,
    ParentSymmetryData,
    ProgressState,
    ScanConfig,
    ScanPlan,
    SinglePhasePlan,
    SingleTrialTask,
)

def normalize_direction(d: List[float], eps: float = 1e-12) -> List[float]:
    """Compat wrapper: OP 方向归一化实现位于 geometry.py。"""
    return _normalize_direction(d, eps=eps)


def _canonicalize_unit_direction(vec, eps: float = 1e-12):
    v = np.asarray(vec, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    v = v / n
    for x in v:
        if abs(float(x)) > eps:
            if float(x) < 0.0:
                v = -v
            break
    v[np.abs(v) < eps] = 0.0
    return v


def _dir_key(v: np.ndarray, ndigits: int = 8):
    return tuple(float(x) for x in np.round(v, ndigits))


def generate_spread_unit_directions(
    dim: int,
    n_random: int,
    rng: np.random.Generator,
    fixed_dirs: Optional[List[List[float]]] = None,
    n_candidates: int = 2000,
    eps: float = 1e-12,
    ndigits: int = 8,
) -> List[List[float]]:
    selected: List[np.ndarray] = []
    seen = set()

    if fixed_dirs:
        for d in fixed_dirs:
            v = _canonicalize_unit_direction(d, eps=eps)
            if v is None:
                continue
            k = _dir_key(v, ndigits=ndigits)
            if k in seen:
                continue
            selected.append(v)
            seen.add(k)

    target_total = len(selected) + max(0, int(n_random))
    if target_total == len(selected):
        return [v.tolist() for v in selected]

    cand: List[np.ndarray] = []
    cand_seen = set()
    max_draws = max(n_candidates * 5, target_total * 200)
    for _ in range(max_draws):
        v = _canonicalize_unit_direction(rng.normal(size=dim), eps=eps)
        if v is None:
            continue
        k = _dir_key(v, ndigits=ndigits)
        if (k in seen) or (k in cand_seen):
            continue
        cand.append(v)
        cand_seen.add(k)
        if len(cand) >= n_candidates:
            break

    while len(selected) < target_total and cand:
        if not selected:
            v = cand.pop(0)
            selected.append(v)
            seen.add(_dir_key(v, ndigits=ndigits))
            continue

        best_idx = None
        best_score = 1e9
        for i, v in enumerate(cand):
            score = max(abs(float(np.dot(v, s))) for s in selected)
            if score < best_score:
                best_score = score
                best_idx = i
                if best_score < 1e-3:
                    break
        if best_idx is None:
            break
        v = cand.pop(best_idx)
        selected.append(v)
        seen.add(_dir_key(v, ndigits=ndigits))

    return [v.tolist() for v in selected]


def build_op_directions_for_combo(
    m: int,
    base_dirs: List[List[float]],
    rng: np.random.Generator,
    derived: DerivedScanConfig,
) -> List[List[float]]:
    if m < 2:
        return [list(d) for d in base_dirs]
    if not derived.enable_random_op_directions or derived.n_random_op_directions_per_combo <= 0:
        return [list(d) for d in base_dirs]
    return generate_spread_unit_directions(
        dim=m,
        n_random=derived.n_random_op_directions_per_combo,
        rng=rng,
        fixed_dirs=base_dirs,
        n_candidates=derived.random_op_direction_candidates,
    )


def _mode_cell_signature(landau_data: LandauBasisData, mode_key: str) -> Tuple[str, Tuple[int, ...], Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]:
    cell = landau_data.mode_cell_by_key.get(mode_key)
    if cell is None:
        return ("parent", (), (), (), ())
    matrix = () if cell.supercell_matrix is None else tuple(int(x) for x in np.asarray(cell.supercell_matrix, dtype=int).reshape(-1))
    lattice = tuple(float(x) for x in np.round(np.asarray(cell.lattice, dtype=float).reshape(-1), 10))
    frac = tuple(float(x) for x in np.round(np.asarray(cell.frac, dtype=float).reshape(-1), 10))
    numbers = tuple(int(x) for x in np.asarray(cell.numbers, dtype=int).reshape(-1))
    return (str(cell.cell_kind), matrix, lattice, frac, numbers)


def _validate_combo_same_cell(landau_data: LandauBasisData, mode_indices: List[Tuple[str, int]], *, source: str) -> None:
    if not mode_indices:
        raise RuntimeError(f"Combo spec from {source} has no modes.")
    signatures = []
    for key, mode_i in mode_indices:
        if key not in landau_data.landau_basis:
            raise RuntimeError(f"Combo spec from {source} references unknown mode key {key!r}.")
        if not (0 <= int(mode_i) < len(landau_data.landau_basis[key])):
            raise RuntimeError(f"Combo spec from {source} has out-of-range mode index: {key}[{mode_i}].")
        signatures.append(_mode_cell_signature(landau_data, key))
    if len(set(signatures)) > 1:
        raise RuntimeError(
            "Combo spec mixes modes from different cells. Explicit mode lifting is not implemented. "
            f"source={source}, mode_indices={mode_indices}"
        )


def auto_generate_combos(landau_data: LandauBasisData, max_size: int, auto_dir_pool_2d: List[List[float]]) -> List[Dict[str, Any]]:
    all_modes: List[Tuple[str, int]] = []
    mode_dims: Dict[Tuple[str, int], int] = {}
    mode_cells: Dict[Tuple[str, int], Tuple[str, Tuple[int, ...], Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]] = {}
    for mode_key, vecs in landau_data.landau_basis.items():
        for i in range(len(vecs)):
            mode = (mode_key, i)
            all_modes.append(mode)
            mode_dims[mode] = int(np.asarray(vecs[i]).size)
            mode_cells[mode] = _mode_cell_signature(landau_data, mode_key)

    combos: List[Dict[str, Any]] = []
    if max_size <= 1:
        return combos

    rejected_different_cell = 0
    for m1, m2 in combinations(all_modes, 2):
        if mode_cells[m1] != mode_cells[m2]:
            rejected_different_cell += 1
            continue
        if mode_dims[m1] != mode_dims[m2]:
            raise RuntimeError(
                "Internal combo consistency error: same-cell modes have different vector lengths. "
                f"modes={m1}, {m2}, dims={mode_dims[m1]}, {mode_dims[m2]}"
            )
        combos.append({
            "target_sgs": set(),
            "mode_indices": [m1, m2],
            "op_directions": [d[:] for d in auto_dir_pool_2d],
        })
    if rejected_different_cell:
        print(f"[COMBO][REJECT] rejected_different_cell_pairs={rejected_different_cell}")
    return combos


def build_combo_specs(
    landau_data: LandauBasisData,
    config: ScanConfig,
) -> Tuple[List[Dict[str, Any]], bool]:
    confirmed_combos = copy.deepcopy(config.confirmed_combos)
    if confirmed_combos:
        for spec_i, raw_spec in enumerate(confirmed_combos):
            mode_indices = [(str(m[0]), int(m[1])) for m in raw_spec["mode_indices"]]
            _validate_combo_same_cell(landau_data, mode_indices, source=f"confirmed_combos[{spec_i}]")
        return confirmed_combos, True

    print("[INFO] confirmed combos 为空，直接使用 AUTO_COMBO 生成策略扫描。")
    combos_auto = auto_generate_combos(
        landau_data,
        max_size=config.auto_combo_max_size,
        auto_dir_pool_2d=config.auto_dir_pool_2d,
    )
    return combos_auto, False


def _normalize_combo_target_sgs(target_sgs: Any) -> Optional[List[int]]:
    if isinstance(target_sgs, set):
        return sorted(int(x) for x in target_sgs)
    if target_sgs is None:
        return None
    return [int(x) for x in target_sgs]


def build_single_phase_plan(landau_data: LandauBasisData, derived: DerivedScanConfig) -> SinglePhasePlan:
    single_strains = list(product(derived.strain_a_grid, derived.strain_b_grid, derived.strain_c_grid))
    basis_sizes = {key: len(landau_data.landau_basis.get(key, [])) for key in landau_data.mode_keys}
    total_tasks = 0
    for key in landau_data.mode_keys:
        total_tasks += basis_sizes[key] * len(derived.amp_grid) * len(single_strains)
    return SinglePhasePlan(
        mode_keys=list(landau_data.mode_keys),
        basis_sizes=basis_sizes,
        amp_grid=list(derived.amp_grid),
        single_strains=[(float(sa), float(sb), float(sc)) for sa, sb, sc in single_strains],
        total_tasks=int(total_tasks),
    )


def build_combo_phase_plan(
    parent_data: ParentStructureData,
    landau_data: LandauBasisData,
    config: ScanConfig,
    derived: DerivedScanConfig,
) -> ComboPhasePlan:
    combo_specs_raw, using_confirmed = build_combo_specs(
        landau_data=landau_data,
        config=config,
    )
    if not combo_specs_raw:
        if not derived.failure_options.empty_combo_scan_continues:
            raise RuntimeError(
                "No combo specs were generated. failure_policy='strict' does not allow an empty combo scan. "
                "Use compatible same-cell modes or choose failure_policy='debug'/'permissive' for diagnostic runs."
            )
        failure_policy_warning(config, "No combo specs were generated; combo planning will continue with an empty spec list.")
    combo_strains = list(product(derived.strain_a_grid, derived.strain_b_grid, derived.strain_c_grid))
    rng = np.random.default_rng(int(config.random_op_direction_seed))

    specs: List[ComboPlanSpec] = []
    total_tasks = 0
    for spec_i, raw_spec in enumerate(combo_specs_raw):
        mode_indices = [(str(m[0]), int(m[1])) for m in raw_spec["mode_indices"]]
        directions = build_op_directions_for_combo(
            m=len(mode_indices),
            base_dirs=[list(d) for d in raw_spec["op_directions"]],
            rng=rng,
            derived=derived,
        )
        combo_target_sgs = _normalize_combo_target_sgs(raw_spec.get("target_sgs"))
        plan_spec = ComboPlanSpec(
            spec_index=int(spec_i),
            mode_indices=mode_indices,
            combo_target_sgs=combo_target_sgs,
            directions=directions,
        )
        specs.append(plan_spec)
        total_tasks += len(directions) * len(derived.rho_grid) * len(combo_strains)

    return ComboPhasePlan(
        specs=specs,
        rho_grid=list(derived.rho_grid),
        combo_strains=[(float(sa), float(sb), float(sc)) for sa, sb, sc in combo_strains],
        using_confirmed=bool(using_confirmed),
        total_tasks=int(total_tasks),
    )


def build_scan_plan(
    parent_data: ParentStructureData,
    symmetry: ParentSymmetryData,
    landau_data: LandauBasisData,
    config: ScanConfig,
    derived: DerivedScanConfig,
    debug: DebugOptions,
) -> ScanPlan:
    """把模式基和派生网格展开成扫描计划。

    输出的 ScanPlan 只描述任务顺序和参数网格，不执行 trial，也不写盘；这让 plan 阶段
    可以单独停下检查 single/combo 数量、combo specs 和随机方向是否符合预期。
    """
    debug_stage_event("plan", "begin", debug, {
        "n_mode_blocks": len(landau_data.mode_keys),
        "strict_pointgroup_match": symmetry.strict_pointgroup_match,
    })
    single_phase = build_single_phase_plan(landau_data, derived)
    combo_phase = build_combo_phase_plan(parent_data, landau_data, config, derived)
    plan = ScanPlan(single_phase=single_phase, combo_phase=combo_phase)
    debug_stage_event("plan", "end", debug, {
        "single_total_tasks": single_phase.total_tasks,
        "combo_total_tasks": combo_phase.total_tasks,
        "combo_specs": len(combo_phase.specs),
        "using_confirmed": combo_phase.using_confirmed,
    })
    return plan



def iter_single_trial_tasks(
    phase_plan: SinglePhasePlan,
    landau_data: LandauBasisData,
    progress_state: ProgressState,
) -> Iterator[SingleTrialTask]:
    mode_keys = phase_plan.mode_keys
    for mode_block_i in range(progress_state.mode_block_i, len(mode_keys)):
        mode_key = mode_keys[mode_block_i]
        n_modes = phase_plan.basis_sizes.get(mode_key, 0)
        if n_modes <= 0:
            continue
        mode_start = progress_state.mode_i if mode_block_i == progress_state.mode_block_i else 0
        for mode_i in range(mode_start, n_modes):
            amp_start = progress_state.amp_i if (mode_block_i == progress_state.mode_block_i and mode_i == progress_state.mode_i) else 0
            for amp_i in range(amp_start, len(phase_plan.amp_grid)):
                amp = float(phase_plan.amp_grid[amp_i])
                strain_start = progress_state.strain_i if (mode_block_i == progress_state.mode_block_i and mode_i == progress_state.mode_i and amp_i == progress_state.amp_i) else 0
                for strain_i in range(strain_start, len(phase_plan.single_strains)):
                    sa, sb, sc = phase_plan.single_strains[strain_i]
                    yield SingleTrialTask(
                        mode_block_i=mode_block_i,
                        mode_i=mode_i,
                        amp_i=amp_i,
                        strain_i=strain_i,
                        mode_key=mode_key,
                        mode_idx=mode_i,
                        amp=amp,
                        sa=float(sa),
                        sb=float(sb),
                        sc=float(sc),
                        n_mode_blocks=len(mode_keys),
                        n_modes=n_modes,
                        n_amps=len(phase_plan.amp_grid),
                        n_strains=len(phase_plan.single_strains),
                    )


def iter_combo_trial_tasks(
    phase_plan: ComboPhasePlan,
    progress_state: ProgressState,
) -> Iterator[ComboTrialTask]:
    for spec_i in range(progress_state.spec_i, len(phase_plan.specs)):
        spec = phase_plan.specs[spec_i]
        dir_start = progress_state.dir_i if spec_i == progress_state.spec_i else 0
        for dir_i in range(dir_start, len(spec.directions)):
            direction = spec.directions[dir_i]
            rho_start = progress_state.rho_i if (spec_i == progress_state.spec_i and dir_i == progress_state.dir_i) else 0
            for rho_i in range(rho_start, len(phase_plan.rho_grid)):
                rho = float(phase_plan.rho_grid[rho_i])
                cstrain_start = progress_state.cstrain_i if (spec_i == progress_state.spec_i and dir_i == progress_state.dir_i and rho_i == progress_state.rho_i) else 0
                for cstrain_i in range(cstrain_start, len(phase_plan.combo_strains)):
                    sa, sb, sc = phase_plan.combo_strains[cstrain_i]
                    yield ComboTrialTask(
                        spec_i=spec_i,
                        dir_i=dir_i,
                        rho_i=rho_i,
                        cstrain_i=cstrain_i,
                        mode_indices=list(spec.mode_indices),
                        op_direction=[float(x) for x in direction],
                        rho=rho,
                        sa=float(sa),
                        sb=float(sb),
                        sc=float(sc),
                        combo_target_sgs=copy.deepcopy(spec.combo_target_sgs),
                        n_specs=len(phase_plan.specs),
                        n_dirs=len(spec.directions),
                        n_rhos=len(phase_plan.rho_grid),
                        n_cstrains=len(phase_plan.combo_strains),
                    )
