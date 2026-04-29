# -*- coding: utf-8 -*-
"""高对称 k 点的 star 和 little group 分析。

坐标约定与 symmetry.py 保持一致：实空间分数坐标按 ``f' = f @ R.T + t`` 变换。
为了保持相位 ``k·f`` 不变，reciprocal fractional row vector 必须使用
``k' = k @ inv(R)``；不能把 reciprocal vector 当成 direct-space vector 处理。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .geometry import wrap_diff, wrap_frac
from .models import KStarData


def normalize_kpoint(k: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """把 reciprocal fractional k 点归一到 [0,1)，并稳定处理接近 0/1 的分量。"""
    out = wrap_frac(np.asarray(k, dtype=float).reshape(3))
    out[np.isclose(out, 0.0, atol=tol)] = 0.0
    out[np.isclose(out, 1.0, atol=tol)] = 0.0
    return wrap_frac(out)


def kpoints_equiv_mod_reciprocal(k1: np.ndarray, k2: np.ndarray, tol: float = 1e-8) -> bool:
    """判断两个 k 点是否相差整数 reciprocal lattice vector。"""
    return bool(np.max(np.abs(wrap_diff(np.asarray(k1, dtype=float) - np.asarray(k2, dtype=float)))) <= tol)


def transform_k_by_rotation(k: np.ndarray, R: np.ndarray, convention: str = "row") -> np.ndarray:
    """按实空间 row-vector convention 变换 k 点。

    当前项目中原子分数坐标由 ``f' = f @ R.T + t`` 变换。相位不变性要求
    ``k' @ R = k``，因此 reciprocal row vector 为 ``k' = k @ inv(R)``。
    """
    if convention != "row":
        raise ValueError(f"Unsupported k transform convention: {convention!r}")
    R_arr = np.asarray(R, dtype=int).reshape(3, 3)
    R_inv_float = np.linalg.inv(R_arr)
    R_inv = np.rint(R_inv_float).astype(int)
    identity = np.eye(3, dtype=int)
    if (
        not np.allclose(R_inv_float, R_inv, atol=1e-8)
        or not np.array_equal(R_arr @ R_inv, identity)
        or not np.array_equal(R_inv @ R_arr, identity)
    ):
        raise RuntimeError(
            "Rotation matrix is not an integer unimodular operation; cannot transform k by inv(R). "
            f"R={R_arr.tolist()}, inv(R)={R_inv_float.tolist()}"
        )
    return normalize_kpoint(np.asarray(k, dtype=float).reshape(3) @ R_inv)


def find_little_group_indices(k: np.ndarray, primitive_rotations: np.ndarray, tol: float = 1e-8) -> List[int]:
    """返回满足 Rk = k + G 的 primitive symmetry operation indices。"""
    out: List[int] = []
    k0 = normalize_kpoint(k, tol=tol)
    for idx, R in enumerate(primitive_rotations):
        if kpoints_equiv_mod_reciprocal(transform_k_by_rotation(k0, R), k0, tol=tol):
            out.append(int(idx))
    return out


def build_k_star(
    k: np.ndarray,
    primitive_rotations: np.ndarray,
    high_symmetry_label: str = "K",
    tol: float = 1e-8,
) -> KStarData:
    """构造 k-star 和 little group。

    k-star 是所有 primitive point-group rotations 作用后的唯一 k arms；little group
    是保持 representative k 到等价 reciprocal lattice vector 的操作集合。
    """
    k0 = normalize_kpoint(k, tol=tol)
    arms: List[np.ndarray] = []
    star_operation_indices: List[int] = []
    for idx, R in enumerate(primitive_rotations):
        arm = transform_k_by_rotation(k0, R)
        if not any(kpoints_equiv_mod_reciprocal(arm, old, tol=tol) for old in arms):
            arms.append(arm)
            star_operation_indices.append(int(idx))

    little = find_little_group_indices(k0, primitive_rotations, tol=tol)
    minus_k = normalize_kpoint(-k0, tol=tol)
    has_minus = any(kpoints_equiv_mod_reciprocal(minus_k, arm, tol=tol) for arm in arms)
    is_gamma = kpoints_equiv_mod_reciprocal(k0, np.zeros(3), tol=tol)

    return KStarData(
        high_symmetry_label=str(high_symmetry_label),
        representative_k=k0,
        arms=arms,
        arm_labels=[f"{high_symmetry_label}_{i}" for i in range(len(arms))],
        little_group_indices=little,
        star_operation_indices=star_operation_indices,
        little_group_size=int(len(little)),
        star_size=int(len(arms)),
        is_gamma=bool(is_gamma),
        has_minus_k_in_star=bool(has_minus),
        diagnostics={
            "tol": float(tol),
            "minus_k": minus_k,
            "k_transform_convention": "row: k @ inv(R)",
        },
    )


def _selfcheck_kpoint_analysis() -> None:
    Rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    kx = np.array([0.5, 0.0, 0.0])
    ky = transform_k_by_rotation(kx, Rz90)
    assert kpoints_equiv_mod_reciprocal(ky, np.array([0.0, 0.5, 0.0]))

    R_hex = np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=int)
    k = np.array([1.0 / 3.0, 0.0, 0.0])
    expected = normalize_kpoint(k @ np.linalg.inv(R_hex))
    transformed = transform_k_by_rotation(k, R_hex)
    wrong = normalize_kpoint(k @ R_hex.T)
    assert kpoints_equiv_mod_reciprocal(transformed, expected)
    assert not kpoints_equiv_mod_reciprocal(transformed, wrong)

    rng = np.random.default_rng(123)
    rotations = [
        Rz90,
        R_hex,
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int),
    ]
    for R in rotations:
        for _ in range(10):
            f = rng.random(3)
            kval = rng.random(3)
            fp = f @ R.T
            kp = kval @ np.linalg.inv(R)
            phase_diff = float(np.dot(kp, fp) - np.dot(kval, f))
            assert abs(phase_diff - round(phase_diff)) < 1e-8
