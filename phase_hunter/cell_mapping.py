# -*- coding: utf-8 -*-
"""input cell -> primitive cell 映射。

seekpath 后端返回 primitive cell 下的 k 点。这里集中处理 input/conventional
cell 中原子到 primitive cell 原子加整数晶格平移的映射。
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .geometry import cart_to_frac, frac_to_cart, pbc_distance, validate_lattice, wrap_frac
from .models import Crystal, PrimitiveCellMapping


def _primitive_match_for_frac(
    frac_unwrapped: np.ndarray,
    z: int,
    primitive: Crystal,
    tol: float,
    origin_shift: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray, float]:
    origin = np.zeros(3, dtype=float) if origin_shift is None else np.asarray(origin_shift, dtype=float).reshape(3)
    frac_corrected = np.asarray(frac_unwrapped, dtype=float).reshape(3) - origin
    best: Tuple[int, np.ndarray, float] | None = None
    for p in np.flatnonzero(np.asarray(primitive.numbers, dtype=int) == int(z)):
        diff = frac_corrected - np.asarray(primitive.frac[int(p)], dtype=float)
        shift = np.rint(diff).astype(int)
        residual = diff - shift
        err = float(np.linalg.norm(residual @ primitive.lattice))
        if best is None or err < best[2]:
            best = (int(p), shift.astype(int), err)
    if best is None:
        raise RuntimeError(f"Primitive mapping failed: no primitive atom with atomic number Z={int(z)}.")
    if best[2] > tol:
        # PBC fallback gives a clearer diagnostic if an alternative image is closer.
        dists = pbc_distance(wrap_frac(frac_corrected), primitive.frac, primitive.lattice)
        raise RuntimeError(
            "Primitive mapping failed: input atom cannot be matched to primitive atom within tolerance. "
            f"Z={int(z)}, best_error={best[2]:.3e}, min_pbc_distance={float(np.min(dists)):.3e}, tol={float(tol):.3e}"
        )
    return best


def _primitive_match_for_cart(
    cart: np.ndarray,
    z: int,
    primitive: Crystal,
    tol: float,
    origin_shift: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray, float]:
    frac_unwrapped = cart_to_frac(np.asarray(cart, dtype=float), primitive.lattice).reshape(3)
    return _primitive_match_for_frac(frac_unwrapped, z, primitive, tol, origin_shift=origin_shift)


def _candidate_origin_shifts(input_frac_in_primitive: np.ndarray, input_numbers: np.ndarray, primitive: Crystal) -> List[np.ndarray]:
    """从同元素 input/primitive 原子配对生成可能的 primitive origin shift。"""
    candidates = [np.zeros(3, dtype=float)]
    for i, frac_i in enumerate(input_frac_in_primitive):
        same = np.flatnonzero(np.asarray(primitive.numbers, dtype=int) == int(input_numbers[i]))
        for p in same:
            candidates.append(wrap_frac(np.asarray(frac_i, dtype=float) - np.asarray(primitive.frac[int(p)], dtype=float)))

    unique: List[np.ndarray] = []
    for cand in candidates:
        if not any(np.max(np.abs(((cand - old + 0.5) % 1.0) - 0.5)) < 1e-8 for old in unique):
            unique.append(np.asarray(cand, dtype=float))
    return unique


def _try_primitive_origin_shift(
    input_frac_in_primitive: np.ndarray,
    input_numbers: np.ndarray,
    primitive: Crystal,
    origin_shift: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    atom_map = np.empty(len(input_numbers), dtype=int)
    shifts = np.empty((len(input_numbers), 3), dtype=int)
    errors: List[float] = []
    primitive_to_input = np.full(primitive.nsites, -1, dtype=int)
    for i, frac_unwrapped in enumerate(input_frac_in_primitive):
        p, shift, err = _primitive_match_for_frac(
            frac_unwrapped,
            int(input_numbers[i]),
            primitive,
            tol,
            origin_shift=origin_shift,
        )
        atom_map[i] = int(p)
        shifts[i] = shift
        errors.append(float(err))
        if primitive_to_input[p] < 0:
            primitive_to_input[p] = int(i)
    if np.any(primitive_to_input < 0):
        return atom_map, shifts, primitive_to_input, float("inf"), origin_shift
    return atom_map, shifts, primitive_to_input, float(max(errors) if errors else 0.0), origin_shift


def _dataset_get(ds: Any, key: str, default: Any = None) -> Any:
    if isinstance(ds, dict):
        return ds.get(key, default)
    return getattr(ds, key, default)


def _spglib_standardized_frac(input_frac: np.ndarray, transformation_matrix: np.ndarray, origin_shift: np.ndarray) -> np.ndarray:
    """把 input fractional 坐标变到 spglib/seekpath standard primitive fractional 框架。

    spglib 的 row-vector 约定在本项目中等价写成：

        f_standard = f_input @ transformation_matrix.T + origin_shift

    该变换会同时处理 input cell 重新取向和标准化；对 156.vasp 这类矩形输入 cell
    到六方 primitive cell 的情况，不能只靠 Cartesian 直接投影或单一 origin shift。
    """
    frac = np.asarray(input_frac, dtype=float)
    P = np.asarray(transformation_matrix, dtype=float).reshape(3, 3)
    origin = np.asarray(origin_shift, dtype=float).reshape(3)
    return frac @ P.T + origin


def _try_spglib_standard_mapping(
    input_crystal: Crystal,
    primitive_crystal: Crystal,
    symprec: float,
    tol: float,
) -> Optional[PrimitiveCellMapping]:
    """优先使用 spglib dataset 的 standardization/mapping_to_primitive 建立映射。

    这条路径用于 backend primitive cell 与 spglib/seekpath 标准 primitive setting 一致的情况。
    如果 dataset 字段缺失或验证误差过大，返回 None，让调用方使用通用匹配路径。
    """
    try:
        import spglib  # type: ignore[import-untyped]
    except Exception:
        return None

    ds = spglib.get_symmetry_dataset(input_crystal.to_spglib_cell(), symprec=float(symprec))
    if ds is None:
        return None
    transformation = _dataset_get(ds, "transformation_matrix")
    origin_shift = _dataset_get(ds, "origin_shift")
    mapping_to_primitive = _dataset_get(ds, "mapping_to_primitive")
    if transformation is None or origin_shift is None or mapping_to_primitive is None:
        return None

    P = np.asarray(transformation, dtype=float).reshape(3, 3)
    origin = np.asarray(origin_shift, dtype=float).reshape(3)
    mapping = np.asarray(mapping_to_primitive, dtype=int).reshape(-1)
    if mapping.shape[0] != input_crystal.nsites:
        return None

    input_lattice = validate_lattice(input_crystal.lattice)
    primitive_lattice = validate_lattice(primitive_crystal.lattice)
    match_tol = max(float(symprec), float(tol), 1e-8) * 5.0

    atom_map = np.empty(input_crystal.nsites, dtype=int)
    shifts = np.empty((input_crystal.nsites, 3), dtype=int)
    errors: List[float] = []
    primitive_to_input = np.full(primitive_crystal.nsites, -1, dtype=int)
    frac_standard = _spglib_standardized_frac(input_crystal.frac, P, origin)

    for i, frac_i in enumerate(frac_standard):
        z = int(input_crystal.numbers[i])
        p_hint = int(mapping[i])
        best: Tuple[int, np.ndarray, float] | None = None

        if 0 <= p_hint < primitive_crystal.nsites and int(primitive_crystal.numbers[p_hint]) == z:
            diff = np.asarray(frac_i, dtype=float) - np.asarray(primitive_crystal.frac[p_hint], dtype=float)
            shift = np.rint(diff).astype(int)
            residual = diff - shift
            err = float(np.linalg.norm(residual @ primitive_lattice))
            best = (p_hint, shift.astype(int), err)

        if best is None or best[2] > match_tol:
            try:
                best = _primitive_match_for_frac(frac_i, z, primitive_crystal, match_tol)
            except RuntimeError:
                return None

        if best[2] > match_tol:
            return None
        atom_map[i] = int(best[0])
        shifts[i] = best[1]
        errors.append(float(best[2]))
        if primitive_to_input[int(best[0])] < 0:
            primitive_to_input[int(best[0])] = int(i)

    if np.any(primitive_to_input < 0):
        return None

    roundtrip_errors: List[float] = []
    input_cart_standard = frac_standard @ primitive_lattice
    for i in range(input_crystal.nsites):
        p = int(atom_map[i])
        frac_image = primitive_crystal.frac[p] + shifts[i]
        cart_back = frac_image @ primitive_lattice
        roundtrip_errors.append(float(np.linalg.norm(cart_back - input_cart_standard[i])))

    return PrimitiveCellMapping(
        input_crystal=input_crystal,
        primitive_crystal=primitive_crystal,
        input_lattice=input_lattice,
        primitive_lattice=primitive_lattice,
        input_atom_to_primitive_atom=atom_map,
        input_atom_to_primitive_lattice_shift=shifts,
        primitive_atom_to_input_atom=primitive_to_input,
        max_atom_mapping_error=float(max(errors) if errors else 0.0),
        max_k_roundtrip_error=float(max(roundtrip_errors) if roundtrip_errors else 0.0),
        diagnostics={
            "input_nsites": int(input_crystal.nsites),
            "primitive_nsites": int(primitive_crystal.nsites),
            "match_tol": float(match_tol),
            "mapping_source": "spglib_standardized_fractional",
            "spglib_transformation_matrix": P.tolist(),
            "spglib_origin_shift": origin.tolist(),
            "spglib_mapping_to_primitive": mapping.tolist(),
            "standardized_fractional_formula": "f_standard = f_input @ transformation_matrix.T + origin_shift",
        },
    )


def build_primitive_mapping_for_kpath(
    input_crystal: Crystal,
    primitive_crystal: Crystal,
    symprec: float,
    tol: float,
) -> PrimitiveCellMapping:
    """构造 input cell 到 primitive cell 的原子映射。

    对每个 input atom i，寻找 primitive atom p 和整数 primitive lattice shift L，使
    r_i ≈ (r_p + L) @ primitive_lattice。匹配严格按元素进行。
    """
    spglib_mapping = _try_spglib_standard_mapping(
        input_crystal=input_crystal,
        primitive_crystal=primitive_crystal,
        symprec=symprec,
        tol=tol,
    )
    if spglib_mapping is not None:
        return spglib_mapping

    input_lattice = validate_lattice(input_crystal.lattice)
    primitive_lattice = validate_lattice(primitive_crystal.lattice)
    input_cart = frac_to_cart(input_crystal.frac, input_lattice)
    input_frac_in_primitive = cart_to_frac(input_cart, primitive_lattice)

    match_tol = max(float(symprec), float(tol), 1e-8) * 5.0
    best: Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None = None
    for origin_shift in _candidate_origin_shifts(input_frac_in_primitive, input_crystal.numbers, primitive_crystal):
        try:
            trial = _try_primitive_origin_shift(
                input_frac_in_primitive,
                input_crystal.numbers,
                primitive_crystal,
                origin_shift,
                match_tol,
            )
        except RuntimeError:
            continue
        if best is None or trial[3] < best[3]:
            best = trial
        if trial[3] <= match_tol:
            break

    if best is None or best[3] > match_tol:
        best_error = float("inf") if best is None else float(best[3])
        raise RuntimeError(
            "Primitive mapping failed: no global primitive origin shift produced a complete same-element mapping. "
            f"best_max_error={best_error:.3e}, tol={float(match_tol):.3e}"
        )
    atom_map, shifts, primitive_to_input, max_error, origin_shift = best

    roundtrip_errors: List[float] = []
    for i in range(input_crystal.nsites):
        p = int(atom_map[i])
        frac_p_image = primitive_crystal.frac[p] + shifts[i] + origin_shift
        cart_back = frac_p_image @ primitive_lattice
        roundtrip_errors.append(float(np.linalg.norm(cart_back - input_cart[i])))

    return PrimitiveCellMapping(
        input_crystal=input_crystal,
        primitive_crystal=primitive_crystal,
        input_lattice=input_lattice,
        primitive_lattice=primitive_lattice,
        input_atom_to_primitive_atom=atom_map,
        input_atom_to_primitive_lattice_shift=shifts,
        primitive_atom_to_input_atom=primitive_to_input,
        max_atom_mapping_error=float(max_error),
        max_k_roundtrip_error=float(max(roundtrip_errors) if roundtrip_errors else 0.0),
        diagnostics={
            "input_nsites": int(input_crystal.nsites),
            "primitive_nsites": int(primitive_crystal.nsites),
            "match_tol": float(match_tol),
            "origin_shift_primitive_frac": origin_shift.tolist(),
        },
    )


def _reciprocal_rows(lattice: np.ndarray) -> np.ndarray:
    """返回不含 2π 的 reciprocal basis rows，满足 k_cart = k_frac @ B。"""
    lat = validate_lattice(lattice)
    return np.linalg.inv(lat).T


def convert_kpoint_between_cells(k: np.ndarray, from_lattice: np.ndarray, to_lattice: np.ndarray) -> np.ndarray:
    """在两个 direct-cell basis 间转换同一个物理 reciprocal vector 的分数坐标。"""
    k_arr = np.asarray(k, dtype=float).reshape(3)
    b_from = _reciprocal_rows(from_lattice)
    b_to = _reciprocal_rows(to_lattice)
    k_cart = k_arr @ b_from
    return np.linalg.solve(b_to.T, k_cart.T).T


def _selfcheck_kpoint_cell_conversion() -> None:
    input_lattice = 2.0 * np.eye(3)
    primitive_lattice = np.eye(3)
    k_input = np.array([1.0, 0.0, 0.0])
    k_primitive = convert_kpoint_between_cells(k_input, input_lattice, primitive_lattice)
    assert np.allclose(k_primitive, np.array([0.5, 0.0, 0.0]))
    k_roundtrip = convert_kpoint_between_cells(k_primitive, primitive_lattice, input_lattice)
    assert np.allclose(k_roundtrip, k_input)
