# -*- coding: utf-8 -*-
"""结构畸变和 strain 操作。

本模块只负责把父相、Landau-like 模式振幅和晶格 strain 组合成 trial 结构；
它不做对称性识别、不写盘、不更新 checkpoint。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import cart_to_frac, frac_to_cart, wrap_frac
from .models import Crystal, ModeCellData


def apply_strain(lattice: np.ndarray, sa: float, sb: float, sc: float) -> np.ndarray:
    """沿三个晶格行向量分别施加缩放 strain。"""
    L = np.array(lattice, float).copy()
    L[0, :] *= float(sa)
    L[1, :] *= float(sb)
    L[2, :] *= float(sc)
    return L


def _mode_cell_fingerprint(cell: ModeCellData) -> Tuple[str, Tuple[int, ...], Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]:
    matrix = cell.supercell_matrix
    if matrix is None:
        matrix_key: Tuple[int, ...] = ()
    else:
        matrix_key = tuple(int(x) for x in np.asarray(matrix, dtype=int).flatten())
    lattice_key = tuple(float(x) for x in np.round(np.asarray(cell.lattice, dtype=float).flatten(), 10))
    frac_key = tuple(float(x) for x in np.round(np.asarray(cell.frac, dtype=float).flatten(), 10))
    numbers_key = tuple(int(x) for x in np.asarray(cell.numbers, dtype=int).flatten())
    return (str(cell.cell_kind), matrix_key, lattice_key, frac_key, numbers_key)


def make_distorted_crystal(
    parent: Crystal,
    landau_basis: Dict[str, List[np.ndarray]],
    mode_amplitudes: Dict[Tuple[str, int], float],
    sa: float,
    sb: float,
    sc: float,
    mode_key_aliases: Optional[Dict[str, str]] = None,
    mode_cell_by_key: Optional[Dict[str, ModeCellData]] = None,
) -> Crystal:
    """根据模式振幅生成 trial 晶体结构。"""
    trial, _disp = make_distorted_crystal_with_displacement(
        parent,
        landau_basis,
        mode_amplitudes,
        sa,
        sb,
        sc,
        mode_key_aliases=mode_key_aliases,
        mode_cell_by_key=mode_cell_by_key,
    )
    return trial


def make_distorted_crystal_with_displacement(
    parent: Crystal,
    landau_basis: Dict[str, List[np.ndarray]],
    mode_amplitudes: Dict[Tuple[str, int], float],
    sa: float,
    sb: float,
    sc: float,
    mode_key_aliases: Optional[Dict[str, str]] = None,
    mode_cell_by_key: Optional[Dict[str, ModeCellData]] = None,
) -> Tuple[Crystal, np.ndarray]:
    """根据模式振幅生成 trial 结构，并返回实际施加的笛卡尔位移数组。

    reduction 需要在写 POSCAR 前知道位移幅度，以避免极小振幅时 spglib 把结构误判回
    高对称父相；因此 evaluator 使用这个函数，兼容 wrapper 仍保留原返回值。
    """
    aliases = mode_key_aliases or {}
    mode_cells = mode_cell_by_key or {}

    resolved_modes: List[Tuple[str, int, float]] = []
    selected_cell_fingerprints = {}
    for (mode_key, midx), amp in mode_amplitudes.items():
        actual_key = mode_key
        if actual_key not in landau_basis and actual_key in aliases:
            actual_key = aliases[actual_key]
        resolved_modes.append((actual_key, int(midx), float(amp)))
        if actual_key in mode_cells:
            selected_cell_fingerprints[actual_key] = _mode_cell_fingerprint(mode_cells[actual_key])

    unique_fingerprints = set(selected_cell_fingerprints.values())
    if len(unique_fingerprints) > 1:
        raise RuntimeError(
            "Cannot mix high-symmetry k-point modes from different supercell cells in one trial: "
            f"{sorted(selected_cell_fingerprints.keys())}"
        )

    base_crystal = mode_cells[next(iter(selected_cell_fingerprints.keys()))].crystal if selected_cell_fingerprints else parent
    N = base_crystal.nsites
    L_new = apply_strain(base_crystal.lattice, sa, sb, sc)
    base_cart = frac_to_cart(base_crystal.frac, L_new)
    disp = np.zeros((N, 3), dtype=float)

    for mode_key, midx, amp in resolved_modes:
        vecs = landau_basis.get(mode_key, None)
        if vecs is None:
            ali = aliases.get(mode_key, None)
            if ali is not None:
                vecs = landau_basis.get(ali, None)
        if vecs is None:
            raise KeyError(f"Mode key '{mode_key}' not found. Available: {list(landau_basis.keys())}")
        if not (0 <= int(midx) < len(vecs)):
            raise IndexError(f"Mode index out of range for {mode_key}: {midx}, n={len(vecs)}")

        if vecs[int(midx)].size != 3 * N:
            if selected_cell_fingerprints and mode_key not in mode_cells:
                raise RuntimeError(
                    "Cannot mix parent-cell Gamma/finite-T modes with high-symmetry k-point supercell modes "
                    "in one trial unless parent-mode lifting to the supercell is implemented. "
                    f"mode={mode_key}[{midx}], mode_dim={vecs[int(midx)].size}, supercell_dim={3 * N}."
                )
            raise ValueError(
                f"Mode length mismatch for {mode_key}[{midx}]: "
                f"mode_dim={vecs[int(midx)].size}, cell_dim={3 * N}. "
                "Check mode_cell_by_key metadata."
            )
        v = vecs[int(midx)].reshape(N, 3)
        disp += float(amp) * v

    cart_new = base_cart + disp
    frac_new = wrap_frac(cart_to_frac(cart_new, L_new))
    trial = Crystal(lattice=L_new, frac=frac_new, numbers=base_crystal.numbers.copy(), symbols=base_crystal.symbols.copy())
    return trial, disp


def _selfcheck_structure_ops_high_k_mode_cells() -> None:
    parent = Crystal(
        lattice=np.eye(3),
        frac=np.array([[0.0, 0.0, 0.0]]),
        numbers=np.array([1], dtype=int),
        symbols=["H"],
    )
    supercell = Crystal(
        lattice=2.0 * np.eye(3),
        frac=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        numbers=np.array([1, 1], dtype=int),
        symbols=["H"],
    )
    mode_cell = ModeCellData(
        cell_kind="high_symmetry_supercell",
        crystal=supercell,
        lattice=supercell.lattice,
        frac=supercell.frac,
        numbers=supercell.numbers,
        symbols=supercell.symbols,
        supercell_matrix=np.diag([2, 1, 1]),
        diagnostics={},
    )
    shifted_cell = ModeCellData(
        cell_kind="high_symmetry_supercell",
        crystal=supercell,
        lattice=supercell.lattice,
        frac=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        numbers=supercell.numbers,
        symbols=supercell.symbols,
        supercell_matrix=np.diag([2, 1, 1]),
        diagnostics={},
    )
    changed_species_cell = ModeCellData(
        cell_kind="high_symmetry_supercell",
        crystal=supercell,
        lattice=supercell.lattice,
        frac=supercell.frac,
        numbers=np.array([1, 2], dtype=int),
        symbols=supercell.symbols,
        supercell_matrix=np.diag([2, 1, 1]),
        diagnostics={},
    )
    assert _mode_cell_fingerprint(mode_cell) != _mode_cell_fingerprint(shifted_cell)
    assert _mode_cell_fingerprint(mode_cell) != _mode_cell_fingerprint(changed_species_cell)
    landau_basis = {
        "Gamma": [np.array([1.0, 0.0, 0.0])],
        "X": [np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])],
    }
    high_k_trial = make_distorted_crystal(
        parent,
        landau_basis,
        {("X", 0): 0.1},
        1.0,
        1.0,
        1.0,
        mode_cell_by_key={"X": mode_cell},
    )
    assert high_k_trial.nsites == 2
    try:
        make_distorted_crystal(
            parent,
            landau_basis,
            {("X", 0): 0.1, ("Gamma", 0): 0.1},
            1.0,
            1.0,
            1.0,
            mode_cell_by_key={"X": mode_cell},
        )
    except RuntimeError as exc:
        assert "Cannot mix parent-cell" in str(exc)
    else:
        raise AssertionError("Expected mixed parent/high-k supercell modes to be rejected.")
