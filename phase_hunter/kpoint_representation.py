# -*- coding: utf-8 -*-
"""primitive-k star Bloch displacement representation。"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .geometry import wrap_diff
from .kpoint_analysis import kpoints_equiv_mod_reciprocal, transform_k_by_rotation
from .models import Crystal
from .symmetry import rotation_cartesian


def find_primitive_atom_mapping_with_lattice_shift(
    primitive_crystal: Crystal,
    R: np.ndarray,
    t: np.ndarray,
    map_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 primitive atom permutation 和整数 lattice shift。

    满足 ``frac_j @ R.T + t ≈ frac_perm[j] + shifts[j]``。匹配严格按元素进行。
    """
    frac = np.asarray(primitive_crystal.frac, dtype=float)
    numbers = np.asarray(primitive_crystal.numbers, dtype=int)
    lattice = np.asarray(primitive_crystal.lattice, dtype=float)
    mapped = frac @ np.asarray(R, dtype=int).T + np.asarray(t, dtype=float).reshape(3)
    n = primitive_crystal.nsites
    perm = np.empty(n, dtype=int)
    shifts = np.empty((n, 3), dtype=int)

    for j in range(n):
        best = None
        for p in np.flatnonzero(numbers == int(numbers[j])):
            diff = mapped[j] - frac[int(p)]
            shift = np.rint(diff).astype(int)
            residual = diff - shift
            # 如果 mapped 在 cell 边界附近，最小像 residual 更稳。
            residual = wrap_diff(residual)
            err = float(np.linalg.norm(residual @ lattice))
            if best is None or err < best[2]:
                best = (int(p), shift.astype(int), err)
        if best is None or best[2] > float(map_tol):
            raise RuntimeError(
                "Cannot find primitive atom mapping with lattice shift for Bloch representation: "
                f"atom={j}, Z={int(numbers[j])}, best_error={best[2] if best else None}, map_tol={float(map_tol):.3e}"
            )
        perm[j] = best[0]
        shifts[j] = best[1]
    if np.unique(perm).size != n:
        raise RuntimeError("Primitive atom mapping is not one-to-one.")
    return perm, shifts


def _find_target_star_arm(k_target: np.ndarray, star_arms: List[np.ndarray], tol: float = 1e-8) -> int:
    for idx, arm in enumerate(star_arms):
        if kpoints_equiv_mod_reciprocal(k_target, arm, tol=tol):
            return int(idx)
    raise RuntimeError(
        "Space-group operation maps k arm outside the supplied star. "
        f"k_target={np.asarray(k_target, dtype=float).tolist()}, "
        f"star={[np.asarray(a, dtype=float).tolist() for a in star_arms]}"
    )


def build_star_bloch_representation_matrix(
    primitive_crystal: Crystal,
    R: np.ndarray,
    t: np.ndarray,
    star_arms: List[np.ndarray],
    map_tol: float,
    phase_convention: str = "cell_periodic_minus",
    k_tol: float = 1e-8,
) -> np.ndarray:
    """构造完整 k-star induced Bloch displacement representation。

    表示空间为 ``direct_sum_a C^(3N)``，每个完整空间群操作把 source arm
    ``k_a`` 映射到 target arm ``k_b = k_a @ inv(R)``。矩阵块采用与单 k
    表示相同的 phase convention：
    ``exp(-2πi k_b·L_j) * R_cart``。
    """
    if phase_convention != "cell_periodic_minus":
        raise ValueError(f"Unsupported phase_convention={phase_convention!r}")
    arms = [np.asarray(a, dtype=float).reshape(3) for a in star_arms]
    if not arms:
        raise ValueError("Cannot build star Bloch representation for an empty k-star.")
    n = primitive_crystal.nsites
    arm_dim = 3 * n
    dim = arm_dim * len(arms)
    D = np.zeros((dim, dim), dtype=complex)
    perm, shifts = find_primitive_atom_mapping_with_lattice_shift(
        primitive_crystal, R, t, map_tol=map_tol,
    )
    R_cart = rotation_cartesian(primitive_crystal.lattice, np.asarray(R, dtype=int))
    for source_arm_i, k_source in enumerate(arms):
        k_target = transform_k_by_rotation(k_source, R)
        target_arm_i = _find_target_star_arm(k_target, arms, tol=k_tol)
        row0 = target_arm_i * arm_dim
        col0 = source_arm_i * arm_dim
        k_phase = arms[target_arm_i]
        for j in range(n):
            jp = int(perm[j])
            phase = np.exp(-2j * np.pi * float(np.dot(k_phase, shifts[j])))
            D[row0 + 3 * jp: row0 + 3 * jp + 3, col0 + 3 * j: col0 + 3 * j + 3] = phase * R_cart
    return D


def _selfcheck_bloch_representation_phase() -> None:
    crys = Crystal(
        lattice=np.eye(3),
        frac=np.array([[0.0, 0.0, 0.0]]),
        numbers=np.array([1]),
        symbols=["H"],
    )
    star = [np.array([0.5, 0.0, 0.0])]
    Ds = build_star_bloch_representation_matrix(
        crys,
        np.eye(3, dtype=int),
        np.array([1.0, 0.0, 0.0]),
        star,
        map_tol=1e-8,
    )
    assert np.allclose(Ds, -np.eye(3))

    Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    star_xy = [np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.5, 0.0])]
    Dxy = build_star_bloch_representation_matrix(
        crys,
        Rz90,
        np.zeros(3),
        star_xy,
        map_tol=1e-8,
    )
    assert Dxy.shape == (6, 6)
    assert np.linalg.norm(Dxy[3:6, 0:3]) > 0.5
