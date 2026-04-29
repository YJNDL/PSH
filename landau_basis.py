# -*- coding: utf-8 -*-
"""Landau-like / symmetry-adapted displacement basis 构造。

这里构造的是对称适配的候选位移模式，用于枚举低对称候选结构；它不等价于直接计算
真实热力学稳定相变路径，模式标签的物理解释应结合后续能量/声子结果。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .cell_mapping import build_primitive_mapping_for_kpath, convert_kpoint_between_cells
from .config import REQUIRED_KPATH_BACKEND, build_failure_policy_options, failure_policy_warning
from .debug_tools import debug_stage_event
from .kpath_backend import get_high_symmetry_kpath
from .kpoint_analysis import build_k_star
from .kpoint_modes import decompose_star_induced_mechanical_representation
from .kpoint_representation import build_star_bloch_representation_matrix
from .kpoint_supercell import (
    build_commensurate_supercell,
    choose_commensurate_supercell_matrix,
    rationalize_kpoint,
    realify_star_complex_modes_to_supercell,
)
from .models import (
    Crystal,
    DebugOptions,
    HighSymmetryKPoint,
    KPointModeMetadata,
    KStarData,
    LandauBasisData,
    ModeBlockMetadata,
    ModeCellData,
    ParentStructureData,
    ParentSymmetryData,
    ScanConfig,
)
from .symmetry import get_symmetry_ops


def _safe_mode_key_part(value: str) -> str:
    """把 k 点/sector 标签转成稳定的内部 key 片段。

    这里生成的是内部 mode ID，不是标准不可约表示名称；高对称 k 标签仍保留原始物理含义。
    """
    text = str(value).strip() or "UNLABELED"
    replacements = {
        "\\Gamma": "G",
        "Γ": "G",
        "+": "plus",
        "-": "minus",
        "/": "_",
        "\\": "_",
        " ": "_",
        ".": "p",
        ",": "_",
        "@": "_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    out = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    compact = "_".join(part for part in "".join(out).split("_") if part)
    return compact or "UNLABELED"


def make_unique_mode_key(base_key: str, existing: Mapping[str, Any]) -> str:
    """生成稳定唯一的内部 mode key；重复时追加 ``__01``、``__02``。"""
    base = _safe_mode_key_part(base_key)
    if base not in existing:
        return base
    idx = 1
    while True:
        candidate = f"{base}__{idx:02d}"
        if candidate not in existing:
            return candidate
        idx += 1

def _extra_high_symmetry_kpoints(config: ScanConfig) -> List[HighSymmetryKPoint]:
    extra = config.extra_target_kpoints_fractional or []
    labels = config.extra_target_kpoint_labels or []
    out: List[HighSymmetryKPoint] = []
    for i, k_raw in enumerate(extra):
        label = str(labels[i]) if i < len(labels) else f"Kextra{i + 1}"
        k = np.asarray(k_raw, dtype=float).reshape(3)
        rat, den, ok = rationalize_kpoint(
            k,
            max_den=int(config.kpoint_rationalize_max_den),
            tol=float(config.kpoint_tol),
        )
        out.append(HighSymmetryKPoint(
            label=label,
            k_fractional=k,
            basis=str(config.extra_target_kpoints_basis),
            source_backend="user_extra",
            path_convention="user",
            is_path_endpoint=False,
            path_segments=[],
            rationalized_k=rat if ok else None,
            denominator=int(den) if ok else None,
            is_commensurate=bool(ok),
            diagnostics={"original_label": label},
        ))
    return out


def _convert_kpoint_to_backend_primitive(
    *,
    kpoint: HighSymmetryKPoint,
    parent: Crystal,
    primitive: Crystal,
    config: ScanConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int], bool, Dict[str, Any]]:
    original = np.asarray(kpoint.k_fractional, dtype=float).reshape(3)
    basis = str(kpoint.basis).strip().lower()
    diagnostics: Dict[str, Any] = {
        "original_k_fractional": original.tolist(),
        "original_k_basis": str(kpoint.basis),
        "conversion_error": None,
    }
    if basis in {"backend_primitive_reciprocal", "primitive", "standard_primitive"}:
        converted = original
    elif basis == "input":
        converted = convert_kpoint_between_cells(original, parent.lattice, primitive.lattice)
    else:
        raise RuntimeError(
            f"Unsupported extra_target_kpoints_basis={kpoint.basis!r}. "
            "Supported: backend_primitive_reciprocal, primitive, standard_primitive, input."
        )
    rat, den, ok = rationalize_kpoint(
        converted,
        max_den=int(config.kpoint_rationalize_max_den),
        tol=float(config.kpoint_tol),
    )
    diagnostics.update({
        "converted_k_backend_primitive": converted.tolist(),
        "rationalized_k": None if not ok else rat.tolist(),
        "denominator": None if not ok else int(den),
    })
    return converted, rat if ok else None, int(den) if ok else None, bool(ok), diagnostics


def _store_kpoint_metadata(
    *,
    key: str,
    kpoint_metadata: Dict[str, KPointModeMetadata],
    kpoint: HighSymmetryKPoint,
    k_vector: np.ndarray,
    k_basis: str,
    k_star: KStarData,
    mode_cell_key: Optional[str],
    complex_available: bool,
    real_available: bool,
    block_index: int,
    block_dimension: int,
    diagnostics: Dict[str, Any],
) -> None:
    kpoint_metadata[key] = KPointModeMetadata(
        internal_mode_key=key,
        high_symmetry_label=kpoint.label,
        original_label=str(kpoint.diagnostics.get("original_label", kpoint.label)),
        k_fractional=np.asarray(k_vector, dtype=float).reshape(3),
        k_basis=str(k_basis),
        k_star=list(k_star.arms),
        little_group_size=int(k_star.little_group_size),
        star_size=int(k_star.star_size),
        block_index=int(block_index),
        block_dimension=int(block_dimension),
        complex_basis_available=bool(complex_available),
        real_supercell_available=bool(real_available),
        mode_cell_key=mode_cell_key,
        phase_convention="cell_periodic_minus",
        realification_strategy=str(diagnostics.get("realification_strategy", "")),
        label_status="not_assigned",
        diagnostics=dict(diagnostics),
    )


def build_high_symmetry_kpoint_landau_context(
    parent_data: ParentStructureData,
    symmetry: ParentSymmetryData,
    config: ScanConfig,
    debug: DebugOptions,
) -> LandauBasisData:
    """构造 high-symmetry primitive-k 位移模式。

    主流程：
    k-path backend -> primitive cell mapping -> k-star/little group -> complex Bloch D_k(g)
    -> full star-induced complex commutant block -> commensurate supercell real modes。
    """
    debug_stage_event("basis", "high_symmetry_k_begin", debug, {
        "backend": REQUIRED_KPATH_BACKEND,
        "selection": config.high_symmetry_kpoint_selection,
        "structure_dimensionality": config.structure_dimensionality,
    })

    kpath = get_high_symmetry_kpath(
        parent_data.crystal,
        symprec=float(config.symprec_parent),
        angle_tolerance=float(config.angle_tolerance),
        backend=REQUIRED_KPATH_BACKEND,
        selection=str(config.high_symmetry_kpoint_selection),
        labels=config.high_symmetry_kpoint_labels,
        include_gamma=bool(config.include_gamma_high_symmetry),
        structure_dimensionality=str(config.structure_dimensionality),
        rationalize_max_den=int(config.kpoint_rationalize_max_den),
        tol=float(config.kpoint_tol),
    )
    selected_kpoints = list(kpath.selected_kpoints)
    selected_kpoints.extend(_extra_high_symmetry_kpoints(config))

    primitive = kpath.primitive_crystal
    _mapping = build_primitive_mapping_for_kpath(
        input_crystal=parent_data.crystal,
        primitive_crystal=primitive,
        symprec=float(config.symprec_parent),
        tol=float(config.kpoint_tol),
    )
    mapping_diagnostics: Dict[str, Any] = dict(_mapping.diagnostics)
    mapping_max_atom_error = float(_mapping.max_atom_mapping_error)
    print(
        "[KPATH][MAPPING] "
        f"source={mapping_diagnostics.get('mapping_source', 'unknown')} "
        f"max_atom_error={mapping_max_atom_error:.3e} "
        f"input_nsites={parent_data.crystal.nsites} primitive_nsites={primitive.nsites}"
    )
    primitive_rotations, primitive_translations = get_symmetry_ops(
        primitive,
        symprec=float(config.symprec_parent),
        angle_tolerance=float(config.angle_tolerance),
    )

    landau_basis: Dict[str, List[np.ndarray]] = {}
    mode_metadata: Dict[str, ModeBlockMetadata] = {}
    mode_cell_by_key: Dict[str, ModeCellData] = {}
    kpoint_metadata: Dict[str, KPointModeMetadata] = {}

    map_tol = max(float(config.symprec_parent) * 5.0, 1e-5)
    if not selected_kpoints:
        raise RuntimeError("No high-symmetry k points were selected; fail-fast mode requires at least one.")

    for kpoint in selected_kpoints:
        k_converted, k_rat, _k_den, k_commensurate, conversion_diag = _convert_kpoint_to_backend_primitive(
            kpoint=kpoint,
            parent=parent_data.crystal,
            primitive=primitive,
            config=config,
        )
        if not k_commensurate or k_rat is None:
            raise RuntimeError(
                f"High-symmetry k point {kpoint.label} is not commensurate within "
                f"max_den={config.kpoint_rationalize_max_den}: k={np.asarray(k_converted, dtype=float).tolist()}"
            )
        k = np.asarray(k_rat if k_rat is not None else k_converted, dtype=float).reshape(3)
        k_star = build_k_star(
            k,
            primitive_rotations,
            high_symmetry_label=kpoint.label,
            tol=float(config.kpoint_tol),
        )
        print(
            "[KPOINT] "
            f"label={kpoint.label} k={np.round(k, 8).tolist()} "
            f"star_size={k_star.star_size} little_group_size={k_star.little_group_size} "
            f"arms={[np.round(a, 8).tolist() for a in k_star.arms]}"
        )

        P = choose_commensurate_supercell_matrix(
            k_star,
            max_den=int(config.kpoint_rationalize_max_den),
            max_size=max(1, int(config.high_symmetry_max_supercell_size) // max(1, primitive.nsites)),
            tol=float(config.kpoint_tol),
            structure_dimensionality=str(config.structure_dimensionality),
        )
        supercell = build_commensurate_supercell(primitive, P, tol=float(config.kpoint_tol))
        print(
            "[KPOINT][SUPERCELL] "
            f"label={kpoint.label} matrix={P.tolist()} det={supercell.det} "
            f"natoms={supercell.supercell_crystal.nsites}"
        )

        Dstar_ops = [
            build_star_bloch_representation_matrix(
                primitive,
                primitive_rotations[int(op_i)],
                primitive_translations[int(op_i)],
                list(k_star.arms),
                map_tol=map_tol,
                k_tol=float(config.kpoint_tol),
            )
            for op_i in range(len(primitive_rotations))
        ]
        blocks = decompose_star_induced_mechanical_representation(
            Dstar_ops,
            seed=int(config.random_op_direction_seed) + 900001,
            tol=float(config.mode_block_tol),
            high_symmetry_label=kpoint.label,
            representative_k=k_star.representative_k,
            star_size=int(k_star.star_size),
        )
        if not blocks:
            raise RuntimeError(f"No full star-induced complex blocks were produced for k point {kpoint.label}.")
        for block in blocks:
            block_number = int(block.block_index) + 1
            base_key = f"KPT_{_safe_mode_key_part(kpoint.label)}_STAR_BLOCK{block_number:03d}"
            occupied: Dict[str, Any] = {}
            occupied.update(landau_basis)
            occupied.update(mode_metadata)
            key = make_unique_mode_key(base_key, occupied)
            base_diagnostics = {
                **conversion_diag,
                "high_symmetry_label_role": "k-path label, not external standard representation name",
                "label_status": "not_assigned",
                "label_note": "Standard representation naming is intentionally disabled in the mode-only build path.",
                "star_induced_representation": True,
                "star_size": int(k_star.star_size),
                "star_arms": [np.asarray(a, dtype=float).tolist() for a in k_star.arms],
                "full_group_operation_count": int(len(Dstar_ops)),
                "little_group_indices": list(k_star.little_group_indices),
                "star_operation_indices": list(k_star.star_operation_indices),
                "primitive_mapping": mapping_diagnostics,
                "output_is_poscar_ready": True,
                "realification_strategy": str(config.real_mode_strategy),
            }
            real_modes = realify_star_complex_modes_to_supercell(
                block,
                list(k_star.arms),
                supercell,
                phase_convention="cell_periodic_minus",
                strategy=str(config.real_mode_strategy),
                tol=float(config.kpoint_tol),
            )
            if not real_modes:
                raise RuntimeError(
                    f"No real supercell modes were produced for k point {kpoint.label} full star block {block_number:03d}."
                )
            landau_basis[key] = real_modes
            mode_cell_by_key[key] = ModeCellData(
                cell_kind="high_symmetry_supercell",
                crystal=supercell.supercell_crystal,
                lattice=supercell.supercell_crystal.lattice,
                frac=supercell.supercell_crystal.frac,
                numbers=supercell.supercell_crystal.numbers,
                symbols=supercell.supercell_crystal.symbols,
                supercell_matrix=supercell.supercell_matrix,
                diagnostics={
                    "primitive_nsites": primitive.nsites,
                    "mapping_max_atom_error": mapping_max_atom_error,
                    "kpath_backend": kpath.backend,
                    "non_diagonal_supercell_supported": True,
                    "star_induced_representation": True,
                },
            )
            _store_kpoint_metadata(
                key=key,
                kpoint_metadata=kpoint_metadata,
                kpoint=kpoint,
                k_vector=k_star.representative_k,
                k_basis="backend_primitive_reciprocal",
                k_star=k_star,
                mode_cell_key=key,
                complex_available=False,
                real_available=True,
                block_index=block_number,
                block_dimension=int(block.dimension_complex),
                diagnostics={
                    **base_diagnostics,
                    "supercell_det": int(supercell.det),
                    "supercell_matrix": supercell.supercell_matrix.tolist(),
                },
            )
            mode_metadata[key] = ModeBlockMetadata(
                key=key,
                source="high_symmetry_kpoint",
                source_kind="supercell_real",
                sector_label=None,
                high_symmetry_label=kpoint.label,
                k_vector=np.asarray(k_star.representative_k, dtype=float).reshape(3),
                k_basis="backend_primitive_reciprocal",
                arm_index=None,
                star_size=int(k_star.star_size),
                little_group_size=int(k_star.little_group_size),
                block_index=block_number,
                block_dimension=int(block.dimension_complex),
                mode_count=len(real_modes),
                is_gamma=bool(k_star.is_gamma),
                cell_kind="high_symmetry_supercell",
                supercell_matrix=supercell.supercell_matrix,
                phase_convention="cell_periodic_minus",
                realification_strategy=str(config.real_mode_strategy),
                label_status="not_assigned",
                diagnostics={
                    **base_diagnostics,
                    "supercell_det": int(supercell.det),
                    "supercell_matrix": supercell.supercell_matrix.tolist(),
                },
            )
            print(
                "[KPOINT-MODE] "
                f"key={key} high_symmetry_label={kpoint.label} "
                f"k={np.round(k_star.representative_k, 8).tolist()} "
                f"supercell={supercell.supercell_matrix.tolist()} "
                f"modes={len(real_modes)} label_status=not_assigned"
            )

    return LandauBasisData(
        landau_basis=landau_basis,
        mode_keys=sorted(landau_basis.keys()),
        mode_metadata=mode_metadata,
        mode_cell_by_key=mode_cell_by_key,
        kpoint_metadata=kpoint_metadata,
    )


def build_landau_context(
    parent_data: ParentStructureData,
    symmetry: ParentSymmetryData,
    config: ScanConfig,
    debug: DebugOptions,
) -> LandauBasisData:
    """构造 Landau-like 位移模式上下文。

    这里的模式是对称适配的候选位移基，用于低对称结构枚举；它不是直接的热力学相变路径
    计算，内部 mode key 更适合做候选结构索引，物理解释应结合后续能量或声子结果。
    """
    debug_stage_event("basis", "begin", debug, {
        "point_group": symmetry.point_group_used,
    })

    print(f"[INFO] 开始构造 high-symmetry primitive-k 模式基... point-group={symmetry.point_group_used}")
    landau_basis: Dict[str, List[np.ndarray]] = {}
    mode_metadata: Dict[str, ModeBlockMetadata] = {}
    mode_cell_by_key: Dict[str, ModeCellData] = {}
    kpoint_metadata: Dict[str, KPointModeMetadata] = {}
    high_k_data = build_high_symmetry_kpoint_landau_context(parent_data, symmetry, config, debug)
    high_keys = set(high_k_data.landau_basis)
    high_keys.update(high_k_data.mode_metadata)
    high_keys.update(high_k_data.mode_cell_by_key)
    high_keys.update(high_k_data.kpoint_metadata)
    occupied: Dict[str, object] = {}
    key_map: Dict[str, str] = {}
    for old_key in sorted(high_keys):
        final_key = make_unique_mode_key(old_key, occupied)
        occupied[final_key] = None
        key_map[old_key] = final_key
        if final_key != old_key:
            print(f"[KEY-REMAP] old={old_key} new={final_key}")

    for old_key, final_key in key_map.items():
        if old_key in high_k_data.landau_basis:
            landau_basis[final_key] = list(high_k_data.landau_basis[old_key])
        if old_key in high_k_data.mode_metadata:
            mode_metadata[final_key] = replace(high_k_data.mode_metadata[old_key], key=final_key)
        if old_key in high_k_data.mode_cell_by_key:
            mode_cell_by_key[final_key] = high_k_data.mode_cell_by_key[old_key]
        if old_key in high_k_data.kpoint_metadata:
            kp_meta = high_k_data.kpoint_metadata[old_key]
            mode_cell_key = final_key if kp_meta.mode_cell_key == old_key else kp_meta.mode_cell_key
            kpoint_metadata[final_key] = replace(kp_meta, internal_mode_key=final_key, mode_cell_key=mode_cell_key)

    if not landau_basis:
        if not build_failure_policy_options(config.failure_policy).empty_high_k_modes_continue:
            raise RuntimeError("No real high-k supercell modes were produced. Fail-fast mode does not allow empty basis output.")
        failure_policy_warning(config, "No real high-k modes were produced; continuing with an empty basis for diagnostics.")

    print("[INFO] Landau basis summary:")
    for key in sorted(landau_basis.keys()):
        print(f"  mode {key}: {len(landau_basis[key])} modes")

    context = LandauBasisData(
        landau_basis=landau_basis,
        mode_keys=sorted(landau_basis.keys()),
        mode_metadata=dict(mode_metadata),
        mode_cell_by_key=dict(mode_cell_by_key),
        kpoint_metadata=dict(kpoint_metadata),
    )
    debug_stage_event("basis", "end", debug, {
        "n_mode_blocks": len(context.mode_keys),
        "n_modes_total": int(sum(len(v) for v in landau_basis.values())),
    })
    return context
