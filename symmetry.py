# -*- coding: utf-8 -*-
"""spglib 对称性识别、点群处理和位移表示矩阵。

find_permutation() 在这里实现：对称操作必须保持元素种类，因此按元素组做周期性
最小像距离的一一全局匹配，比逐原子贪心更适合同元素近简并位置。
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .debug_tools import debug_stage_event
from .geometry import pbc_distance_matrix, wrap_frac
from .models import Crystal, DebugOptions, ParentStructureData, ParentSymmetryData, ScanConfig


def _require_spglib():
    try:
        import spglib  # noqa: F401
    except Exception as e:
        raise RuntimeError("This script requires spglib. Install: pip install spglib") from e


def get_spglib_dataset(crys: Crystal, symprec: float, angle_tolerance: float):
    _require_spglib()
    import spglib  # type: ignore

    cell = crys.to_spglib_cell()
    if angle_tolerance is not None and angle_tolerance > 0:
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    else:
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:
        raise RuntimeError(
            "spglib failed to identify the parent symmetry dataset. "
            "Please check the POSCAR quality or adjust SYMPREC_PARENT."
        )
    return ds


def dataset_get(ds: Any, key: str, default=None):
    if isinstance(ds, dict):
        return ds.get(key, default)
    return getattr(ds, key, default)


def identify_spacegroup(crys: Crystal, symprec: float, angle_tolerance: float) -> Tuple[int, Optional[str], Optional[str]]:
    ds = get_spglib_dataset(crys, symprec=symprec, angle_tolerance=angle_tolerance)
    sg_num = int(dataset_get(ds, "number", -1))
    sg_symbol = dataset_get(ds, "international", None)
    pg = dataset_get(ds, "pointgroup", None)
    return sg_num, sg_symbol, pg


def get_symmetry_ops(crys: Crystal, symprec: float, angle_tolerance: float):
    _require_spglib()
    import spglib  # type: ignore

    cell = crys.to_spglib_cell()
    if angle_tolerance is not None and angle_tolerance > 0:
        symm = spglib.get_symmetry(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    else:
        symm = spglib.get_symmetry(cell, symprec=symprec)
    if symm is None:
        raise RuntimeError(
            "spglib failed to generate symmetry operations. "
            "Please check the POSCAR quality or adjust SYMPREC_PARENT."
        )
    rotations = np.array(symm["rotations"], int)
    translations = np.array(symm["translations"], float)
    return rotations, translations


def normalize_pointgroup_symbol(pg: Optional[str]) -> Optional[str]:
    if pg is None:
        return None
    s = str(pg).strip().replace("−", "-")
    s = s.replace(" ", "").lower()
    if s.startswith("mm") and len(s) == 3 and s[2].isdigit() and s[2] in {"4", "6"}:
        s = f"{s[2]}mm"
    if s == "3mm":
        s = "3m"
    return s

def resolve_parent_symmetry(parent_data: ParentStructureData, config: ScanConfig, debug: DebugOptions) -> ParentSymmetryData:
    """识别父相对称性并确定实际使用的点群。

    spglib 给出空间群、点群和对称操作；STRICT_POINTGROUP_MATCH 用于确保用户 hint 与
    spglib 识别结果一致，避免在错误点群下继续构造表示矩阵和模式基。
    """
    debug_stage_event("symmetry", "begin", debug, {
        "symprec_parent": config.symprec_parent,
        "angle_tolerance": config.angle_tolerance,
    })

    sg_parent, sg_sym, pg_raw = identify_spacegroup(
        parent_data.crystal,
        symprec=config.symprec_parent,
        angle_tolerance=config.angle_tolerance,
    )
    pg = normalize_pointgroup_symbol(pg_raw)
    pg_hint = normalize_pointgroup_symbol(config.parent_point_group_hint)
    pg_use = pg_hint or pg

    print(f"[INFO] spglib 识别的父相 space-group: {sg_parent} ({sg_sym})")
    print(f"[INFO] spglib 识别的父相 point-group: {pg_raw} -> {pg} (用户 hint: {config.parent_point_group_hint} -> {pg_hint})")

    if pg_use is None:
        raise RuntimeError(
            "无法从 spglib 识别 point-group，且未提供 PARENT_POINT_GROUP_HINT。\n"
            "请设置 PARENT_POINT_GROUP_HINT (例如 'mm2','4mm','-3m','6/mmm' 等)。"
        )
    if bool(config.strict_pointgroup_match) and pg_hint is not None and pg is not None and pg_hint != pg:
        raise RuntimeError(
            "STRICT_POINTGROUP_MATCH=True, but PARENT_POINT_GROUP_HINT does not match spglib result: "
            f"hint={pg_hint}, spglib={pg}. "
            "请检查父相 POSCAR、SYMPREC_PARENT 或关闭 STRICT_POINTGROUP_MATCH 后再运行。"
        )

    rotations, translations = get_symmetry_ops(
        parent_data.crystal,
        symprec=config.symprec_parent,
        angle_tolerance=config.angle_tolerance,
    )
    symmetry = ParentSymmetryData(
        spacegroup_number=int(sg_parent),
        spacegroup_symbol=sg_sym,
        point_group_raw=pg_raw,
        point_group_normalized=pg,
        point_group_hint=pg_hint,
        point_group_used=str(pg_use),
        rotations=rotations,
        translations=translations,
        strict_pointgroup_match=bool(config.strict_pointgroup_match),
    )
    debug_stage_event("symmetry", "end", debug, {
        "spacegroup_number": symmetry.spacegroup_number,
        "spacegroup_symbol": symmetry.spacegroup_symbol,
        "point_group_used": symmetry.point_group_used,
        "n_sym_ops": int(len(symmetry.rotations)),
        "strict_pointgroup_match": symmetry.strict_pointgroup_match,
    })
    return symmetry

def _atom_group_cost_matrix(mapped_frac_group: np.ndarray, target_frac_group: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    # 周期性最小像距离：target - mapped 先 wrap 到 [-0.5, 0.5)，再转笛卡尔长度。
    return pbc_distance_matrix(mapped_frac_group, target_frac_group, lattice)


def _require_linear_sum_assignment():
    """只使用 SciPy 的 assignment solver；不再保留 NumPy fallback。"""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SciPy is required for robust atom matching. "
            "Please install it with `pip install scipy` in the active Python environment."
        ) from e
    return linear_sum_assignment


def _raise_atom_group_match_error(
    element_label: int,
    group_size: int,
    cost: np.ndarray,
    matched_dist: np.ndarray,
    map_tol: float,
) -> None:
    max_match_dist = float(np.max(matched_dist)) if matched_dist.size else float("nan")
    failed = matched_dist[matched_dist > map_tol]
    if failed.size:
        min_failure_dist = float(np.min(failed))
    else:
        invalid = cost[cost > map_tol]
        min_failure_dist = float(np.min(invalid)) if invalid.size else float("nan")

    raise RuntimeError(
        f"Cannot find one-to-one atom mapping for element Z={int(element_label)} "
        f"(group_size={int(group_size)}). "
        f"max_match_dist={max_match_dist:.3e}, "
        f"min_failure_dist={min_failure_dist:.3e}, "
        f"map_tol={float(map_tol):.3e}. "
        f"Try increasing SYMPREC_PARENT."
    )


def _match_atom_group(
    mapped_frac_group: np.ndarray,
    target_frac_group: np.ndarray,
    lattice: np.ndarray,
    map_tol: float,
    element_label: int,
) -> np.ndarray:
    cost = _atom_group_cost_matrix(mapped_frac_group, target_frac_group, lattice)
    n_src, n_tgt = cost.shape
    if n_src != n_tgt:
        raise RuntimeError(
            f"Cannot match element Z={int(element_label)}: "
            f"source_count={int(n_src)}, target_count={int(n_tgt)}."
        )
    if n_src == 0:
        return np.empty(0, dtype=int)
    if not np.all(np.isfinite(cost)):
        _raise_atom_group_match_error(element_label, n_src, cost, np.empty(0, dtype=float), map_tol)

    # 超过 map_tol 的边不直接删除，而是加大惩罚后参与全局匹配；最终再统一校验容差。
    feasible = cost <= map_tol
    scale = max(float(np.max(cost)), abs(float(map_tol)), 1.0)
    invalid_penalty = scale * float(n_src + 1) + 1.0
    assignment_cost = np.where(feasible, cost, invalid_penalty + cost)

    row_ind, col_ind = _require_linear_sum_assignment()(assignment_cost)

    row_ind = np.asarray(row_ind, dtype=int)
    col_ind = np.asarray(col_ind, dtype=int)
    if (
        row_ind.size != n_src
        or col_ind.size != n_src
        or np.unique(row_ind).size != n_src
        or np.unique(col_ind).size != n_src
    ):
        _raise_atom_group_match_error(element_label, n_src, cost, np.empty(0, dtype=float), map_tol)

    local_perm = np.empty(n_src, dtype=int)
    local_perm[row_ind] = col_ind
    matched_dist = cost[np.arange(n_src), local_perm]
    if matched_dist.size and float(np.max(matched_dist)) > map_tol:
        _raise_atom_group_match_error(element_label, n_src, cost, matched_dist, map_tol)

    return local_perm


def find_permutation(crys: Crystal, R: np.ndarray, t: np.ndarray, map_tol: float) -> np.ndarray:
    """返回对称操作下的原子置换 perm[i] = j。

    对称操作必须保持元素种类，因此先按原子序数分组，再在组内做周期性最小像距离的一一
    全局匹配。局部贪心在同元素近简并位置上可能早早占用错误目标，导致后续原子无解。
    """
    frac = crys.frac
    lattice = crys.lattice
    Z = crys.numbers
    N = crys.nsites

    mapped = wrap_frac(frac @ R.T + t)
    perm = np.empty(N, dtype=int)

    # 只在同元素组内匹配，避免跨元素最近邻造成物理上无效的原子映射。
    for z in sorted(int(x) for x in np.unique(Z)):
        group = np.flatnonzero(Z == z)
        local_perm = _match_atom_group(mapped[group], frac[group], lattice, map_tol, element_label=z)
        perm[group] = group[local_perm]

    return perm


def _selfcheck_find_permutation_matching() -> None:
    lattice = np.eye(3, dtype=float)

    crys = Crystal(
        lattice=lattice,
        frac=np.array([
            [0.10, 0.00, 0.00],
            [0.40, 0.00, 0.00],
            [0.20, 0.20, 0.00],
        ], dtype=float),
        numbers=np.array([6, 6, 8], dtype=int),
        symbols=["C", "O"],
    )
    perm = find_permutation(crys, np.eye(3, dtype=int), np.zeros(3, dtype=float), map_tol=1e-8)
    assert np.array_equal(perm, np.array([0, 1, 2], dtype=int))

    mapped = np.array([
        [0.10, 0.00, 0.00],
        [0.91, 0.00, 0.00],
    ], dtype=float)
    target = np.array([
        [0.00, 0.00, 0.00],
        [0.21, 0.00, 0.00],
    ], dtype=float)
    local_perm = _match_atom_group(mapped, target, lattice, map_tol=0.12, element_label=6)
    assert np.array_equal(local_perm, np.array([1, 0], dtype=int))

    rows, cols = _require_linear_sum_assignment()(np.array([
        [0.10, 0.11],
        [0.09, 10.0],
    ], dtype=float))
    assert np.array_equal(rows, np.array([0, 1], dtype=int))
    assert np.array_equal(cols, np.array([1, 0], dtype=int))

    try:
        _match_atom_group(mapped, target, lattice, map_tol=0.08, element_label=6)
    except RuntimeError as e:
        msg = str(e)
        assert "Z=6" in msg and "map_tol" in msg and "SYMPREC_PARENT" in msg
    else:
        raise AssertionError("Expected RuntimeError for infeasible atom mapping.")


def rotation_cartesian(lattice: np.ndarray, R_frac: np.ndarray) -> np.ndarray:
    A = lattice.T
    A_inv = np.linalg.inv(A)
    return A @ R_frac @ A_inv


def build_representation_matrix(crys: Crystal, R: np.ndarray, t: np.ndarray, map_tol: float) -> np.ndarray:
    """构造位移表示矩阵 D。

    D 把第 i 个原子的笛卡尔位移按原子置换和旋转映射到第 j 个原子，用于后续从对称操作
    的 commutant 中提取对称适配位移模式。
    """
    N = crys.nsites
    dim = 3 * N
    D = np.zeros((dim, dim), dtype=float)

    perm = find_permutation(crys, R, t, map_tol=map_tol)
    R_cart = rotation_cartesian(crys.lattice, R)

    for i in range(N):
        j = int(perm[i])
        D[3 * j: 3 * j + 3, 3 * i: 3 * i + 3] = R_cart

    return D
