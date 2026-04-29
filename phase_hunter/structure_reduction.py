# -*- coding: utf-8 -*-
"""畸变结构输出 cell reduction。

本模块只在 trial 位移全部施加完成之后、写 POSCAR 之前使用。它尝试寻找
distorted structure 自身的 primitive cell，并通过原子重构验证；不会把非 Γ
高对称点畸变结构强行压回父相 primitive cell。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import cart_to_frac, frac_to_cart, lattice_volume, wrap_frac
from .models import Crystal, ReducedStructureData
from .periodic_table import Z_TO_SYMBOL


def _symbols_for_numbers_like(original_symbols: List[str], numbers: np.ndarray) -> List[str]:
    """按原结构 symbols 顺序为 reduced numbers 生成 POSCAR header symbols。"""
    present = set(int(z) for z in np.asarray(numbers, dtype=int))
    out: List[str] = []
    for original_sym in original_symbols:
        z = None
        for z_try, sym_try in Z_TO_SYMBOL.items():
            if sym_try == original_sym:
                z = int(z_try)
                break
        if z is not None and z in present and original_sym not in out:
            out.append(original_sym)
    for z in np.asarray(numbers, dtype=int):
        inferred_sym = Z_TO_SYMBOL.get(int(z))
        if inferred_sym is None:
            raise ValueError(f"Cannot infer element symbol for atomic number Z={int(z)}.")
        inferred_sym = str(inferred_sym)
        if inferred_sym not in out:
            out.append(inferred_sym)
    return out


def _reduction_data(
    *,
    crystal: Crystal,
    reduced: Optional[Crystal],
    attempted: bool,
    successful: bool,
    backend: str,
    symprec: float,
    angle_tolerance: float,
    max_reconstruction_error: Optional[float],
    formula_preserved: bool,
    reason: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> ReducedStructureData:
    original_volume = lattice_volume(crystal.lattice)
    if reduced is None:
        reduced_volume = original_volume
        volume_ratio = 1.0
        reduced_nsites = crystal.nsites
    else:
        reduced_volume = lattice_volume(reduced.lattice)
        volume_ratio = original_volume / reduced_volume if reduced_volume > 0.0 else float("inf")
        reduced_nsites = reduced.nsites
    return ReducedStructureData(
        reduction_attempted=bool(attempted),
        reduction_successful=bool(successful),
        backend=str(backend),
        original_nsites=int(crystal.nsites),
        reduced_nsites=int(reduced_nsites),
        original_volume=float(original_volume),
        reduced_volume=float(reduced_volume),
        volume_ratio=float(volume_ratio),
        original_lattice=np.asarray(crystal.lattice, dtype=float).copy(),
        reduced_lattice=None if reduced is None else np.asarray(reduced.lattice, dtype=float).copy(),
        original_frac=np.asarray(crystal.frac, dtype=float).copy(),
        reduced_frac=None if reduced is None else np.asarray(reduced.frac, dtype=float).copy(),
        reduced_numbers=None if reduced is None else np.asarray(reduced.numbers, dtype=int).copy(),
        reduced_symbols=None if reduced is None else list(reduced.symbols),
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
        max_reconstruction_error=max_reconstruction_error,
        formula_preserved=bool(formula_preserved),
        reason=str(reason),
        diagnostics=dict(diagnostics or {}),
    )


def _crystal_from_spglib_tuple(data: Tuple[Any, Any, Any], original_symbols: List[str]) -> Crystal:
    lattice, frac, numbers = data
    numbers_arr = np.asarray(numbers, dtype=int)
    return Crystal(
        lattice=np.asarray(lattice, dtype=float),
        frac=wrap_frac(np.asarray(frac, dtype=float)),
        numbers=numbers_arr,
        symbols=_symbols_for_numbers_like(original_symbols, numbers_arr),
    )


def _call_spglib_find_primitive(crystal: Crystal, symprec: float, angle_tolerance: float) -> Optional[Crystal]:
    import spglib  # type: ignore[import-untyped]

    cell = crystal.to_spglib_cell()
    try:
        result = spglib.find_primitive(cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
    except TypeError:
        result = spglib.find_primitive(cell, symprec=float(symprec))
    if result is None:
        return None
    return _crystal_from_spglib_tuple(result, crystal.symbols)


def _call_spglib_standardize_primitive(crystal: Crystal, symprec: float, angle_tolerance: float) -> Optional[Crystal]:
    import spglib  # type: ignore[import-untyped]

    cell = crystal.to_spglib_cell()
    try:
        result = spglib.standardize_cell(
            cell,
            to_primitive=True,
            no_idealize=True,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )
    except TypeError:
        result = spglib.standardize_cell(
            cell,
            to_primitive=True,
            no_idealize=True,
            symprec=float(symprec),
        )
    if result is None:
        return None
    return _crystal_from_spglib_tuple(result, crystal.symbols)


def _reduce_with_spglib(crystal: Crystal, symprec: float, angle_tolerance: float) -> Crystal:
    """用 spglib 在 distorted structure 上寻找 primitive cell，优先不 idealize。"""
    try:
        reduced = _call_spglib_find_primitive(crystal, symprec, angle_tolerance)
        if reduced is None:
            reduced = _call_spglib_standardize_primitive(crystal, symprec, angle_tolerance)
    except Exception as exc:
        raise RuntimeError(f"spglib distorted-cell reduction failed: {exc}") from exc
    if reduced is None:
        raise RuntimeError("spglib did not find a primitive cell for the distorted structure.")
    return reduced


def _formula_counts(numbers: np.ndarray) -> Counter[int]:
    return Counter(int(z) for z in np.asarray(numbers, dtype=int))


def verify_reduced_reconstructs_original(
    original: Crystal,
    reduced: Crystal,
    tol: float,
) -> Tuple[bool, float, Dict[str, Any]]:
    """验证 reduced cell 的周期平移像可以重构 original distorted supercell。

    验证按元素匹配：每个 original atom 必须能映射到某个 reduced atom 的整数平移像；
    每个 reduced atom 应出现 volume_ratio_int 次。
    """
    diagnostics: Dict[str, Any] = {}
    old_volume = lattice_volume(original.lattice)
    new_volume = lattice_volume(reduced.lattice)
    if new_volume <= 0.0:
        return False, float("inf"), {"reason": "reduced lattice volume is non-positive"}

    ratio = old_volume / new_volume
    ratio_int = int(round(ratio))
    diagnostics["volume_ratio"] = float(ratio)
    diagnostics["volume_ratio_int"] = int(ratio_int)
    if ratio_int < 1 or abs(ratio - ratio_int) > max(float(tol), 1e-6):
        diagnostics["reason"] = "volume ratio is not close to an integer"
        return False, float("inf"), diagnostics
    if int(original.nsites) != ratio_int * int(reduced.nsites):
        diagnostics["reason"] = "atom count ratio does not match integer volume ratio"
        return False, float("inf"), diagnostics

    original_counts = _formula_counts(original.numbers)
    reduced_counts = _formula_counts(reduced.numbers)
    expected_counts = Counter({z: int(count) * ratio_int for z, count in reduced_counts.items()})
    diagnostics["original_counts"] = {str(k): int(v) for k, v in original_counts.items()}
    diagnostics["expected_counts"] = {str(k): int(v) for k, v in expected_counts.items()}
    if original_counts != expected_counts:
        diagnostics["reason"] = "chemical composition is not preserved by reduced cell"
        return False, float("inf"), diagnostics

    reduced_numbers = np.asarray(reduced.numbers, dtype=int)
    reduced_frac = np.asarray(reduced.frac, dtype=float)
    reduced_lattice = np.asarray(reduced.lattice, dtype=float)
    original_frac_in_reduced = np.array([
        cart_to_frac(frac_to_cart(frac_i, original.lattice), reduced_lattice).reshape(3)
        for frac_i in np.asarray(original.frac, dtype=float)
    ], dtype=float)

    origin_candidates: List[np.ndarray] = [np.zeros(3, dtype=float)]
    for frac_i, z_raw in zip(original_frac_in_reduced, np.asarray(original.numbers, dtype=int)):
        for j in np.flatnonzero(reduced_numbers == int(z_raw)):
            origin_candidates.append(wrap_frac(frac_i - reduced_frac[int(j)]))

    unique_origins: List[np.ndarray] = []
    for origin in origin_candidates:
        if not any(np.max(np.abs(((origin - old + 0.5) % 1.0) - 0.5)) < 1e-8 for old in unique_origins):
            unique_origins.append(np.asarray(origin, dtype=float))

    best_failure: Dict[str, Any] = {"max_error": float("inf")}
    for origin_shift in unique_origins:
        counts_per_reduced_atom = np.zeros(reduced.nsites, dtype=int)
        max_error = 0.0
        matched_reduced_indices: List[int] = []
        failed = False
        failure_reason = ""
        failure_atom: Optional[int] = None
        failure_error = float("inf")

        for atom_i, (frac_in_reduced, z_raw) in enumerate(zip(original_frac_in_reduced, np.asarray(original.numbers, dtype=int))):
            candidates = np.flatnonzero(reduced_numbers == int(z_raw))
            best_j: Optional[int] = None
            best_err = float("inf")
            for j in candidates:
                diff = frac_in_reduced - (reduced_frac[int(j)] + origin_shift)
                shift = np.rint(diff).astype(int)
                residual = diff - shift
                err = float(np.linalg.norm(residual @ reduced_lattice))
                if err < best_err:
                    best_err = err
                    best_j = int(j)
            if best_j is None:
                failed = True
                failure_reason = f"no reduced atom candidate for original atom {atom_i} with Z={int(z_raw)}"
                failure_atom = int(atom_i)
                break
            if best_err > float(tol):
                failed = True
                failure_reason = "original atom cannot be reconstructed from reduced cell within tolerance"
                failure_atom = int(atom_i)
                failure_error = float(best_err)
                break
            counts_per_reduced_atom[best_j] += 1
            matched_reduced_indices.append(best_j)
            max_error = max(max_error, best_err)

        if failed:
            if failure_error < float(best_failure.get("max_error", float("inf"))):
                best_failure = {
                    "reason": failure_reason,
                    "failed_atom": failure_atom,
                    "failed_atom_error": failure_error,
                    "origin_shift": origin_shift.tolist(),
                    "max_error": failure_error,
                }
            continue

        if not np.all(counts_per_reduced_atom == ratio_int):
            count_error = float(np.max(np.abs(counts_per_reduced_atom - ratio_int)))
            if count_error < float(best_failure.get("max_error", float("inf"))):
                best_failure = {
                    "reason": "not every reduced atom appears the expected number of times",
                    "origin_shift": origin_shift.tolist(),
                    "counts_per_reduced_atom": [int(x) for x in counts_per_reduced_atom],
                    "max_error": count_error,
                }
            continue

        diagnostics["counts_per_reduced_atom"] = [int(x) for x in counts_per_reduced_atom]
        diagnostics["matched_reduced_indices"] = matched_reduced_indices
        diagnostics["origin_shift"] = origin_shift.tolist()
        diagnostics["n_origin_shift_candidates"] = int(len(unique_origins))
        diagnostics["reason"] = "ok"
        return True, float(max_error), diagnostics

    diagnostics.update(best_failure)
    diagnostics["n_origin_shift_candidates"] = int(len(unique_origins))
    if "reason" not in diagnostics:
        diagnostics["reason"] = "no origin shift candidate reconstructed the original cell"
    return False, float(best_failure.get("max_error", float("inf"))), diagnostics


def reduce_distorted_structure(
    crystal: Crystal,
    displacement: Optional[np.ndarray],
    *,
    backend: str,
    symprec: float,
    angle_tolerance: float,
    verify: bool,
    verify_tol: float,
    min_volume_ratio: float,
    skip_if_amplitude_below: float,
    strict: bool,
) -> Tuple[Crystal, ReducedStructureData]:
    """尝试把 distorted structure 输出为更小 primitive cell。

    返回的 Crystal 只用于 POSCAR 输出；不会修改模式基、trial 顺序或后续 checkpoint。
    """
    backend_norm = str(backend).strip().lower()
    if backend_norm != "spglib":
        raise ValueError(f"Only spglib distorted reduction backend is supported in fail-fast mode; got {backend!r}")

    diagnostics: Dict[str, Any] = {}
    max_disp = None
    if displacement is not None:
        disp_arr = np.asarray(displacement, dtype=float)
        if disp_arr.size:
            max_disp = float(np.max(np.linalg.norm(disp_arr.reshape(-1, 3), axis=1)))
        else:
            max_disp = 0.0
        diagnostics["max_displacement_norm"] = float(max_disp)
        diagnostics["skip_threshold"] = float(skip_if_amplitude_below)
        if max_disp < float(skip_if_amplitude_below):
            data = _reduction_data(
                crystal=crystal,
                reduced=None,
                attempted=False,
                successful=False,
                backend=backend_norm,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
                max_reconstruction_error=None,
                formula_preserved=True,
                reason="Skipped reduction because displacement amplitude is too small; spglib may restore parent symmetry.",
                diagnostics={**diagnostics, "skipped_due_to_small_amplitude": True},
            )
            return crystal, data
    else:
        diagnostics["max_displacement_norm"] = None

    reduced: Optional[Crystal] = None
    used_backend = "spglib"
    errors: List[str] = []
    try:
        reduced = _reduce_with_spglib(crystal, float(symprec), float(angle_tolerance))
    except Exception as exc:
        errors.append(f"spglib: {exc}")

    if reduced is None:
        reason = "Reduction backend failed: " + "; ".join(errors)
        if strict:
            raise RuntimeError(reason)
        data = _reduction_data(
            crystal=crystal,
            reduced=None,
            attempted=True,
            successful=False,
            backend=used_backend,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            max_reconstruction_error=None,
            formula_preserved=True,
            reason=reason,
            diagnostics={**diagnostics, "backend_errors": errors},
        )
        return crystal, data

    original_volume = lattice_volume(crystal.lattice)
    reduced_volume = lattice_volume(reduced.lattice)
    volume_ratio = original_volume / reduced_volume if reduced_volume > 0.0 else float("inf")
    diagnostics["candidate_volume_ratio"] = float(volume_ratio)
    if reduced.nsites >= crystal.nsites:
        reason = "Reduced candidate does not have fewer atoms than original distorted cell."
        if strict:
            raise RuntimeError(reason)
        data = _reduction_data(
            crystal=crystal,
            reduced=reduced,
            attempted=True,
            successful=False,
            backend=used_backend,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            max_reconstruction_error=None,
            formula_preserved=True,
            reason=reason,
            diagnostics=diagnostics,
        )
        return crystal, data
    if volume_ratio < float(min_volume_ratio):
        reason = (
            f"Reduced candidate volume ratio {volume_ratio:.6g} is below "
            f"minimum {float(min_volume_ratio):.6g}."
        )
        if strict:
            raise RuntimeError(reason)
        data = _reduction_data(
            crystal=crystal,
            reduced=reduced,
            attempted=True,
            successful=False,
            backend=used_backend,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            max_reconstruction_error=None,
            formula_preserved=True,
            reason=reason,
            diagnostics=diagnostics,
        )
        return crystal, data

    formula_preserved = True
    max_error: Optional[float] = None
    verify_diag: Dict[str, Any] = {}
    if verify:
        ok, max_error, verify_diag = verify_reduced_reconstructs_original(crystal, reduced, tol=float(verify_tol))
        diagnostics["verification"] = verify_diag
        formula_preserved = "chemical composition" not in str(verify_diag.get("reason", ""))
        if not ok:
            reason = f"Verification failed: {verify_diag.get('reason', 'unknown')}"
            if strict:
                raise RuntimeError(reason)
            data = _reduction_data(
                crystal=crystal,
                reduced=reduced,
                attempted=True,
                successful=False,
                backend=used_backend,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
                max_reconstruction_error=max_error,
                formula_preserved=formula_preserved,
                reason=reason,
                diagnostics=diagnostics,
            )
            return crystal, data

    data = _reduction_data(
        crystal=crystal,
        reduced=reduced,
        attempted=True,
        successful=True,
        backend=used_backend,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        max_reconstruction_error=max_error,
        formula_preserved=formula_preserved,
        reason="reduction successful",
        diagnostics=diagnostics,
    )
    return reduced, data


def _selfcheck_structure_reduction() -> None:
    """轻量自检：验证重构匹配和小振幅跳过逻辑。"""
    reduced = Crystal(
        lattice=np.diag([2.0, 1.0, 1.0]),
        frac=np.array([[0.0, 0.05, 0.0], [0.5, 0.95, 0.0]], dtype=float),
        numbers=np.array([1, 1], dtype=int),
        symbols=["H"],
    )
    original = Crystal(
        lattice=np.diag([2.0, 2.0, 1.0]),
        frac=np.array([
            [0.0, 0.025, 0.0],
            [0.5, 0.975, 0.0],
            [0.0, 0.525, 0.0],
            [0.5, 0.475, 0.0],
        ], dtype=float),
        numbers=np.array([1, 1, 1, 1], dtype=int),
        symbols=["H"],
    )
    ok, err, diag = verify_reduced_reconstructs_original(original, reduced, tol=1e-8)
    if not ok:
        raise AssertionError(f"Expected reduced cell to reconstruct original, err={err}, diag={diag}")

    trial, data = reduce_distorted_structure(
        original,
        np.zeros((original.nsites, 3), dtype=float),
        backend="spglib",
        symprec=1e-5,
        angle_tolerance=-1.0,
        verify=True,
        verify_tol=1e-5,
        min_volume_ratio=1.05,
        skip_if_amplitude_below=1e-4,
        strict=False,
    )
    if trial.nsites != original.nsites or data.reduction_attempted:
        raise AssertionError("Small-displacement reduction should be skipped and keep original cell.")

    doubled = Crystal(
        lattice=np.diag([2.0, 1.0, 1.0]),
        frac=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float),
        numbers=np.array([1, 1], dtype=int),
        symbols=["H"],
    )
    reduced_trial, reduced_data = reduce_distorted_structure(
        doubled,
        np.ones((doubled.nsites, 3), dtype=float) * 0.01,
        backend="spglib",
        symprec=1e-5,
        angle_tolerance=-1.0,
        verify=True,
        verify_tol=1e-5,
        min_volume_ratio=1.05,
        skip_if_amplitude_below=0.0,
        strict=True,
    )
    if not reduced_data.reduction_successful or reduced_trial.nsites != 1:
        raise AssertionError("Expected doubled test cell to reduce to one atom.")
