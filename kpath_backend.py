# -*- coding: utf-8 -*-
"""高对称 k-path 后端封装。

本模块只负责像能带计算一样识别 primitive Brillouin zone 的 special k points/path。
高对称 k 点 label（G/X/M/R/K/...）只表示 k-path convention，不是空间群标准表示名称。
fail-fast 主流程固定使用 SeeK-path；不再按环境自动切换其他 backend。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .kpoint_supercell import rationalize_kpoint
from .models import Crystal, HighSymmetryKPathData, HighSymmetryKPoint
from .periodic_table import Z_TO_SYMBOL

SEEKPATH_PATH_CONVENTION = "hpkot"


def _normalize_k_label(label: str) -> str:
    raw = str(label).strip()
    if raw in {"\\Gamma", "Gamma", "GAMMA", "Γ"}:
        return "G"
    return raw.replace("\\", "")


def _segments_from_path(path_obj: List[List[str]] | List[Tuple[str, str]] | List[str]) -> List[Tuple[str, str]]:
    segments: List[Tuple[str, str]] = []
    for item in path_obj:
        if isinstance(item, str):
            labels = list(item)
        else:
            labels = [str(x) for x in item]
        for a, b in zip(labels[:-1], labels[1:]):
            segments.append((_normalize_k_label(a), _normalize_k_label(b)))
    return segments


def _selected_labels(
    point_coords: Dict[str, np.ndarray],
    path: List[Tuple[str, str]],
    selection: str,
    labels: Optional[List[str]],
    include_gamma: bool,
) -> List[str]:
    selection_norm = str(selection).strip().lower()
    if selection_norm == "labels":
        if not labels:
            raise ValueError("high_symmetry_kpoint_selection='labels' requires high_symmetry_kpoint_labels.")
        selected = [_normalize_k_label(x) for x in labels]
    elif selection_norm == "all_point_coords":
        selected = sorted(point_coords.keys())
    elif selection_norm == "path_endpoints":
        seen = []
        for a, b in path:
            for x in (a, b):
                if x not in seen:
                    seen.append(x)
        selected = seen
    else:
        raise ValueError(f"Unknown high_symmetry_kpoint_selection={selection!r}")

    if include_gamma and "G" in point_coords and "G" not in selected:
        selected.insert(0, "G")
    if not include_gamma:
        selected = [x for x in selected if x != "G"]
    missing = [x for x in selected if x not in point_coords]
    if missing:
        raise RuntimeError(f"Requested high-symmetry k labels not found from backend: {missing}")
    return selected


def _normalize_dimensionality(value: str) -> str:
    dim = str(value).strip().lower()
    if dim in {"2d", "2", "two-dimensional", "two_dimensional"}:
        return "2d"
    if dim in {"3d", "3", "three-dimensional", "three_dimensional"}:
        return "3d"
    raise ValueError(f"Unknown structure_dimensionality={value!r}; use '2d' or '3d'.")


def _kz_distance_to_zero(k: np.ndarray) -> float:
    kz = float(np.asarray(k, dtype=float).reshape(3)[2])
    return abs(((kz + 0.5) % 1.0) - 0.5)


def _filter_labels_by_dimensionality(
    selected: List[str],
    point_coords: Dict[str, np.ndarray],
    *,
    selection: str,
    labels: Optional[List[str]],
    structure_dimensionality: str,
    tol: float,
) -> Tuple[List[str], List[str]]:
    """按结构维度过滤 high-symmetry k 点。

    对二维材料，当前模式生成只处理面内 primitive-BZ 点，因此保留 kz=0 的 special
    points，过滤 A/L/H 这类 out-of-plane 点。用户显式指定 labels 且包含被过滤点时
    直接报错，避免 silently 使用错误维度的 k 点。
    """
    dim = _normalize_dimensionality(structure_dimensionality)
    if dim == "3d":
        return selected, []

    kept: List[str] = []
    removed: List[str] = []
    for label in selected:
        k = point_coords[label]
        if _kz_distance_to_zero(k) <= tol:
            kept.append(label)
        else:
            removed.append(label)

    requested = {_normalize_k_label(x) for x in (labels or [])}
    explicit_removed = [label for label in removed if label in requested]
    if str(selection).strip().lower() == "labels" and explicit_removed:
        details = ", ".join(f"{label}: k={np.asarray(point_coords[label], dtype=float).tolist()}" for label in explicit_removed)
        raise RuntimeError(
            "Requested out-of-plane high-symmetry k labels for structure_dimensionality='2d': "
            f"{details}. Use --structure-dimensionality 3d if this is intentional."
        )
    return kept, removed


def _build_kpath_data(
    *,
    backend: str,
    convention: str,
    primitive_crystal: Crystal,
    raw_point_coords: Dict[str, np.ndarray],
    raw_path: List[Tuple[str, str]],
    selection: str,
    labels: Optional[List[str]],
    include_gamma: bool,
    structure_dimensionality: str,
    rationalize_max_den: int,
    tol: float,
    diagnostics: Dict[str, object],
) -> HighSymmetryKPathData:
    point_coords: Dict[str, np.ndarray] = {}
    original_by_norm: Dict[str, str] = {}
    for raw_label, coord in raw_point_coords.items():
        label = _normalize_k_label(raw_label)
        point_coords[label] = np.asarray(coord, dtype=float).reshape(3)
        original_by_norm[label] = str(raw_label)

    path = [(_normalize_k_label(a), _normalize_k_label(b)) for a, b in raw_path]
    selected = _selected_labels(point_coords, path, selection, labels, include_gamma)
    selected, removed_by_dim = _filter_labels_by_dimensionality(
        selected,
        point_coords,
        selection=selection,
        labels=labels,
        structure_dimensionality=structure_dimensionality,
        tol=tol,
    )
    if not selected:
        raise RuntimeError(
            "No high-symmetry k labels remain after selection/dimensionality filtering. "
            f"selection={selection!r}, structure_dimensionality={structure_dimensionality!r}, "
            f"removed_by_dim={removed_by_dim}."
        )
    selected_points: List[HighSymmetryKPoint] = []
    endpoint_set = {x for seg in path for x in seg}
    for label in selected:
        k = point_coords[label]
        rat, den, ok = rationalize_kpoint(k, max_den=rationalize_max_den, tol=tol)
        selected_points.append(HighSymmetryKPoint(
            label=label,
            k_fractional=k,
            basis="backend_primitive_reciprocal",
            source_backend=backend,
            path_convention=convention,
            is_path_endpoint=label in endpoint_set,
            path_segments=[seg for seg in path if label in seg],
            rationalized_k=rat if ok else None,
            denominator=int(den) if ok else None,
            is_commensurate=bool(ok),
            diagnostics={
                "original_label": original_by_norm.get(label, label),
                "rationalize_tol": float(tol),
                "structure_dimensionality": _normalize_dimensionality(structure_dimensionality),
            },
        ))

    if removed_by_dim:
        print(
            "[KPATH][DIM-FILTER] "
            f"structure_dimensionality={_normalize_dimensionality(structure_dimensionality)} "
            f"removed={' '.join(removed_by_dim)}"
        )
    print(
        "[KPATH] "
        f"backend={backend} convention={convention} "
        f"labels={' '.join(p.label for p in selected_points)} "
        f"path={' | '.join(f'{a}-{b}' for a, b in path)} "
        f"primitive_nsites={primitive_crystal.nsites} selected={selection}"
    )
    return HighSymmetryKPathData(
        backend=backend,
        path_convention=convention,
        primitive_crystal=primitive_crystal,
        point_coords=point_coords,
        path=path,
        selected_kpoints=selected_points,
        diagnostics={
            **diagnostics,
            "original_labels": original_by_norm,
            "structure_dimensionality": _normalize_dimensionality(structure_dimensionality),
            "dimensionality_removed_labels": list(removed_by_dim),
        },
    )


def _get_kpath_seekpath(
    crystal: Crystal,
    symprec: float,
    angle_tolerance: float,
    selection: str,
    labels: Optional[List[str]],
    include_gamma: bool,
    structure_dimensionality: str,
    rationalize_max_den: int,
    tol: float,
) -> HighSymmetryKPathData:
    try:
        import seekpath  # type: ignore[import-not-found,import-untyped]
    except Exception as e:
        raise RuntimeError("seekpath is required for high-symmetry k-path generation. Install seekpath.") from e
    cell = (
        np.asarray(crystal.lattice, dtype=float),
        np.asarray(crystal.frac, dtype=float),
        np.asarray(crystal.numbers, dtype=int),
    )
    result = seekpath.get_path(
        cell,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    raw_points = {str(k): np.asarray(v, dtype=float) for k, v in result["point_coords"].items()}
    raw_path = [(_normalize_k_label(a), _normalize_k_label(b)) for a, b in result["path"]]
    primitive_lattice = np.asarray(result["primitive_lattice"], dtype=float)
    primitive_frac = np.asarray(result["primitive_positions"], dtype=float) % 1.0
    primitive_numbers = np.asarray(result["primitive_types"], dtype=int)
    symbols: List[str] = []
    for z in primitive_numbers:
        sym = Z_TO_SYMBOL.get(int(z), str(int(z)))
        if sym not in symbols:
            symbols.append(sym)
    primitive_crystal = Crystal(primitive_lattice, primitive_frac, primitive_numbers, symbols)
    return _build_kpath_data(
        backend="seekpath",
        convention=SEEKPATH_PATH_CONVENTION,
        primitive_crystal=primitive_crystal,
        raw_point_coords=raw_points,
        raw_path=raw_path,
        selection=selection,
        labels=labels,
        include_gamma=include_gamma,
        structure_dimensionality=structure_dimensionality,
        rationalize_max_den=rationalize_max_den,
        tol=tol,
        diagnostics={"seekpath_keys": sorted(str(k) for k in result.keys())},
    )


def get_high_symmetry_kpath(
    crystal: Crystal,
    symprec: float,
    angle_tolerance: float,
    backend: str = "seekpath",
    selection: str = "path_endpoints",
    labels: Optional[List[str]] = None,
    include_gamma: bool = True,
    structure_dimensionality: str = "3d",
    rationalize_max_den: int = 24,
    tol: float = 1e-8,
) -> HighSymmetryKPathData:
    """识别 primitive BZ 高对称 k 点和路径。

    fail-fast 主流程只允许 seekpath。backend 不是 ``seekpath`` 时立即报错，
    不根据环境自动切换后端。
    """
    backend_norm = str(backend).strip().lower()
    if backend_norm != "seekpath":
        raise RuntimeError(
            "Only seekpath backend is supported in fail-fast mode. "
            f"Received backend={backend!r}."
        )
    return _get_kpath_seekpath(
        crystal,
        symprec,
        angle_tolerance,
        selection,
        labels,
        include_gamma,
        structure_dimensionality,
        rationalize_max_den,
        tol,
    )


def _selfcheck_seekpath_backend_policy() -> None:
    bad_backend = "auto"
    try:
        get_high_symmetry_kpath(
            Crystal(np.eye(3), np.array([[0.0, 0.0, 0.0]]), np.array([1]), ["H"]),
            symprec=1e-5,
            angle_tolerance=-1.0,
            backend=bad_backend,
        )
    except RuntimeError as exc:
        assert "Only seekpath backend" in str(exc)
    else:
        raise AssertionError(f"backend={bad_backend!r} should fail in fail-fast mode.")
