# -*- coding: utf-8 -*-
"""commensurate k 点的 supercell 选择与 complex mode 实化。

这里处理的是能被有限 supercell 折叠的 k 点。非 Γ 模式实化时不强制减去所有原子的
平均位移，避免破坏 Bloch 相位生成的交替位移图样。
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm
from collections import deque
from typing import List, Tuple

import numpy as np

from .geometry import cart_to_frac, frac_to_cart, wrap_frac
from .models import CommensurateSupercell, ComplexModeBlock, Crystal, KStarData


def rationalize_kpoint(k: np.ndarray, max_den: int, tol: float) -> Tuple[np.ndarray, int, bool]:
    """把 k 分量有理化，返回 rationalized k、公共分母和是否在 tol 内成功。"""
    vals = []
    den = 1
    ok = True
    for x in np.asarray(k, dtype=float).reshape(3):
        frac = Fraction(float(x)).limit_denominator(int(max_den))
        val = float(frac.numerator) / float(frac.denominator)
        if abs(float(x) - val) > float(tol):
            ok = False
        vals.append(val % 1.0)
        den = lcm(int(den), int(frac.denominator))
    return np.array(vals, dtype=float), int(den), bool(ok)


def _matrix_det_int(P: np.ndarray) -> int:
    det = int(round(float(np.linalg.det(P))))
    if det <= 0:
        raise ValueError(f"Supercell matrix must have positive determinant, got det={det}.")
    return det


def _is_integer_vector(x: np.ndarray, tol: float) -> bool:
    return bool(np.max(np.abs(np.asarray(x, dtype=float) - np.rint(x))) <= tol)


def _validate_commensurate_matrix(P: np.ndarray, arms: List[np.ndarray], tol: float) -> None:
    for arm in arms:
        if not _is_integer_vector(np.asarray(P, dtype=int).T @ np.asarray(arm, dtype=float), tol=tol):
            raise ValueError(f"Supercell matrix does not fold k arm {np.asarray(arm).tolist()} to Gamma.")


def choose_commensurate_supercell_matrix(
    k_star: KStarData,
    max_den: int,
    max_size: int,
    tol: float,
    structure_dimensionality: str,
) -> np.ndarray:
    """选择满足 ``P.T @ k ∈ Z^3`` 的主路径 supercell 矩阵。

    3D 结构使用 isotropic LCM: ``diag(den, den, den)``。
    2D slab 使用面内 LCM: ``diag(den, den, 1)``，避免无意义放大真空方向。
    """
    arms = list(k_star.arms)
    den = 1
    for arm in arms:
        _, d, ok = rationalize_kpoint(arm, max_den=max_den, tol=tol)
        if not ok:
            raise ValueError(f"k arm cannot be rationalized within max_den={max_den}: {arm.tolist()}")
        den = lcm(int(den), int(d))
    dim = str(structure_dimensionality).strip().lower()
    if dim in {"2d", "2", "two-dimensional", "two_dimensional"}:
        P = np.diag([int(max(1, den)), int(max(1, den)), 1]).astype(int)
    elif dim in {"3d", "3", "three-dimensional", "three_dimensional"}:
        P = np.eye(3, dtype=int) * int(max(1, den))
    else:
        raise ValueError(f"Unknown structure_dimensionality={structure_dimensionality!r}; use '2d' or '3d'.")
    det = _matrix_det_int(P)
    if det > int(max_size):
        raise ValueError(f"Chosen supercell det={det} exceeds high_symmetry_max_supercell_size={int(max_size)}.")
    _validate_commensurate_matrix(P, arms, tol=tol)
    return P


def _coset_key(m: np.ndarray, P_inv: np.ndarray, tol: float) -> Tuple[int, int, int]:
    frac = wrap_frac(np.asarray(m, dtype=float) @ P_inv)
    scale = max(1, int(round(1.0 / max(float(tol), 1e-12))))
    vals = [int(round(float(x) * scale)) % scale for x in frac]
    return (vals[0], vals[1], vals[2])


def enumerate_integer_coset_representatives(P: np.ndarray, tol: float = 1e-10) -> List[np.ndarray]:
    """枚举 ``Z^3 / Z^3 P`` 的整数 coset representatives。

    项目 supercell convention 为 ``A_s = P @ A_p``，primitive shift 的等价关系是
    row-vector ``m ~ m + n P``。用 ``wrap_frac(m @ inv(P))`` 作为 coset key，
    通过 BFS 从原点扩展 ±e_i，直到找到 ``abs(det(P))`` 个不同 cosets。
    """
    P_int = np.asarray(P, dtype=int).reshape(3, 3)
    det = _matrix_det_int(P_int)
    if np.array_equal(P_int, np.diag(np.diag(P_int))):
        diag = [int(x) for x in np.diag(P_int)]
        if any(x <= 0 for x in diag):
            raise ValueError(f"Diagonal supercell entries must be positive: {diag}")
        return [
            np.array([i, j, k], dtype=int)
            for i in range(diag[0])
            for j in range(diag[1])
            for k in range(diag[2])
        ]
    P_inv = np.linalg.inv(P_int.astype(float))
    steps = [
        np.array([1, 0, 0], dtype=int),
        np.array([-1, 0, 0], dtype=int),
        np.array([0, 1, 0], dtype=int),
        np.array([0, -1, 0], dtype=int),
        np.array([0, 0, 1], dtype=int),
        np.array([0, 0, -1], dtype=int),
    ]
    start = np.zeros(3, dtype=int)
    queue: deque[np.ndarray] = deque([start])
    visited_points = {tuple(start.tolist())}
    reps: List[np.ndarray] = []
    seen_cosets = set()
    max_visits = max(128, det * 512)
    while queue and len(reps) < det and len(visited_points) <= max_visits:
        m = queue.popleft()
        key = _coset_key(m, P_inv, tol=tol)
        if key not in seen_cosets:
            seen_cosets.add(key)
            reps.append(m.copy())
        for step in steps:
            nxt = m + step
            nkey = tuple(int(x) for x in nxt.tolist())
            if nkey not in visited_points:
                visited_points.add(nkey)
                queue.append(nxt)
    if len(reps) != det:
        raise RuntimeError(
            f"Failed to enumerate integer cosets for supercell matrix P={P_int.tolist()}; "
            f"expected {det}, got {len(reps)}."
        )
    reps.sort(key=lambda v: (int(np.dot(v, v)), tuple(int(x) for x in v.tolist())))
    return [np.asarray(v, dtype=int) for v in reps]


def build_commensurate_supercell(
    primitive_crystal: Crystal,
    supercell_matrix: np.ndarray,
    tol: float,
) -> CommensurateSupercell:
    """按一般整数矩阵 P 构造 supercell。"""
    P = np.asarray(supercell_matrix, dtype=int)
    det = _matrix_det_int(P)
    translations = enumerate_integer_coset_representatives(P, tol=float(tol))
    if len(translations) != det:
        raise RuntimeError("Internal supercell translation enumeration mismatch.")

    super_lattice = P @ np.asarray(primitive_crystal.lattice, dtype=float)
    frac_rows: List[np.ndarray] = []
    numbers: List[int] = []
    prim_atom: List[int] = []
    prim_shift: List[np.ndarray] = []
    for shift in translations:
        cart = frac_to_cart(np.asarray(primitive_crystal.frac, dtype=float) + shift, primitive_crystal.lattice)
        frac_s = wrap_frac(cart_to_frac(cart, super_lattice))
        for j in range(primitive_crystal.nsites):
            frac_rows.append(frac_s[j])
            numbers.append(int(primitive_crystal.numbers[j]))
            prim_atom.append(int(j))
            prim_shift.append(shift.copy())

    super_crystal = Crystal(
        lattice=super_lattice,
        frac=np.array(frac_rows, dtype=float),
        numbers=np.array(numbers, dtype=int),
        symbols=list(primitive_crystal.symbols),
    )
    if super_crystal.nsites != det * primitive_crystal.nsites:
        raise RuntimeError(
            f"Supercell atom count mismatch: got {super_crystal.nsites}, expected {det * primitive_crystal.nsites}."
        )
    seen_atoms = set()
    for z, f in zip(super_crystal.numbers, super_crystal.frac):
        key = (int(z), tuple(int(round(float(x) / max(float(tol), 1e-10))) for x in wrap_frac(f)))
        if key in seen_atoms:
            raise RuntimeError(f"Duplicate atom generated in supercell for P={P.tolist()}: Z={int(z)}, frac={f.tolist()}")
        seen_atoms.add(key)

    return CommensurateSupercell(
        primitive_crystal=primitive_crystal,
        supercell_crystal=super_crystal,
        supercell_matrix=P,
        det=int(det),
        primitive_translations=[t.astype(int) for t in translations],
        supercell_atom_to_primitive_atom=np.array(prim_atom, dtype=int),
        supercell_atom_to_primitive_lattice_shift=np.array(prim_shift, dtype=int),
        diagnostics={
            "tol": float(tol),
            "n_super_atoms": int(super_crystal.nsites),
            "non_diagonal_supercell_supported": True,
        },
    )


def normalize_mode_for_kpoint(mode: np.ndarray, nsites: int, is_gamma: bool, eps: float = 1e-12) -> np.ndarray:
    """归一化 k-point mode；只有 Γ 模式才去 acoustic 平均位移。"""
    v = np.asarray(mode, dtype=float).reshape(int(nsites), 3).copy()
    if is_gamma:
        v -= np.mean(v, axis=0, keepdims=True)
    norms = np.linalg.norm(v, axis=1)
    maxn = float(np.max(norms)) if norms.size else 0.0
    if maxn < eps:
        return np.zeros((3 * int(nsites),), dtype=float)
    v /= maxn
    flat = v.reshape(-1)
    for x in flat:
        if abs(float(x)) > eps:
            if float(x) < 0.0:
                flat = -flat
            break
    return flat


def realify_star_complex_modes_to_supercell(
    complex_block: ComplexModeBlock,
    star_arms: List[np.ndarray],
    supercell: CommensurateSupercell,
    phase_convention: str,
    strategy: str = "cos_sin",
    tol: float = 1e-8,
) -> List[np.ndarray]:
    """把 full star-induced complex mode 实化到 commensurate supercell。

    对 supercell atom ``(j, L)`` 使用
    ``U[j,L] = sum_a u[a,j] exp(2πi k_a·L)``。这是真正的 star-induced
    表示空间实化路径；不同于 arm-resolved 模式，它一次使用全部 star arms。
    """
    if strategy != "cos_sin":
        raise ValueError(f"Unsupported real_mode_strategy={strategy!r}")
    if phase_convention != "cell_periodic_minus":
        raise ValueError(f"Unsupported phase_convention={phase_convention!r}")
    arms = [np.asarray(a, dtype=float).reshape(3) for a in star_arms]
    if not arms:
        raise ValueError("Cannot realify star mode without star arms.")
    basis = np.asarray(complex_block.basis_complex, dtype=complex)
    n_prim = int(supercell.primitive_crystal.nsites)
    arm_dim = 3 * n_prim
    expected = arm_dim * len(arms)
    if basis.shape[0] != expected:
        raise ValueError(
            f"Star complex block length {basis.shape[0]} does not match 3*N*star_size={expected}."
        )
    out: List[np.ndarray] = []
    n_super = int(supercell.supercell_crystal.nsites)
    is_gamma = all(np.max(np.abs(np.asarray(a))) < tol for a in arms)
    for col in range(basis.shape[1]):
        u_star = basis[:, col].reshape(len(arms), n_prim, 3)
        real_disp = np.zeros((n_super, 3), dtype=float)
        imag_disp = np.zeros((n_super, 3), dtype=float)
        for sidx, (pidx, shift) in enumerate(zip(
            supercell.supercell_atom_to_primitive_atom,
            supercell.supercell_atom_to_primitive_lattice_shift,
        )):
            val = np.zeros(3, dtype=complex)
            for arm_i, k in enumerate(arms):
                phase = np.exp(2j * np.pi * float(np.dot(k, shift)))
                val += u_star[arm_i, int(pidx)] * phase
            real_disp[int(sidx)] = np.real(val)
            imag_disp[int(sidx)] = np.imag(val)
        real_mode = normalize_mode_for_kpoint(real_disp.reshape(-1), n_super, is_gamma=bool(is_gamma))
        if float(np.linalg.norm(real_mode)) > tol:
            out.append(real_mode)
        imag_mode = normalize_mode_for_kpoint(imag_disp.reshape(-1), n_super, is_gamma=bool(is_gamma))
        if float(np.linalg.norm(imag_mode)) > tol:
            out.append(imag_mode)
    return out


def _selfcheck_kpoint_supercell() -> None:
    for k, expected in [
        (np.array([0.5, 0.0, 0.0]), 2),
        (np.array([0.5, 0.5, 0.0]), 2),
        (np.array([0.5, 0.5, 0.5]), 2),
        (np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]), 3),
    ]:
        rat, den, ok = rationalize_kpoint(k, max_den=12, tol=1e-8)
        assert ok and den == expected, (k, rat, den)

    primitive = Crystal(
        lattice=np.eye(3),
        frac=np.array([[0.0, 0.0, 0.0]]),
        numbers=np.array([1], dtype=int),
        symbols=["H"],
    )

    def check_pattern(k: np.ndarray, expected_sign) -> None:
        from .models import KStarData

        k_star = KStarData(
            high_symmetry_label="K",
            representative_k=k,
            arms=[k],
            arm_labels=["K"],
            little_group_indices=[0],
            star_operation_indices=[0],
            little_group_size=1,
            star_size=1,
            is_gamma=False,
            has_minus_k_in_star=True,
            diagnostics={},
        )
        P = choose_commensurate_supercell_matrix(k_star, 12, 64, 1e-8, "3d")
        supercell = build_commensurate_supercell(primitive, P, tol=1e-8)
        block = ComplexModeBlock(
            high_symmetry_label="K",
            k_vector=k,
            little_group_indices=[0],
            basis_complex=np.array([[1.0 + 0j], [0.0 + 0j], [0.0 + 0j]]),
            dimension_complex=1,
            character=np.array([1.0 + 0j]),
            block_index=0,
            diagnostics={},
        )
        modes = realify_star_complex_modes_to_supercell(block, [k], supercell, "cell_periodic_minus")
        mode = modes[0].reshape(supercell.supercell_crystal.nsites, 3)[:, 0]
        by_shift = {
            tuple(int(x) for x in shift): float(mode[i])
            for i, shift in enumerate(supercell.supercell_atom_to_primitive_lattice_shift)
        }
        for shift, sign in expected_sign.items():
            assert np.sign(by_shift[shift]) == sign, (k, shift, by_shift[shift])

    check_pattern(np.array([0.5, 0.0, 0.0]), {(0, 0, 0): 1, (1, 0, 0): -1, (0, 1, 0): 1})
    check_pattern(np.array([0.5, 0.5, 0.0]), {(1, 0, 0): -1, (0, 1, 0): -1, (0, 0, 1): 1})
    check_pattern(np.array([0.5, 0.5, 0.5]), {(1, 0, 0): -1, (0, 1, 0): -1, (0, 0, 1): -1})

    k3 = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
    k_star3 = KStarData("K", k3, [k3], ["K"], [0], [0], 1, 1, False, False, {})
    P3 = choose_commensurate_supercell_matrix(k_star3, 12, 64, 1e-8, "2d")
    assert np.array_equal(P3, np.diag([3, 3, 1]))
    sc3 = build_commensurate_supercell(primitive, P3, tol=1e-8)
    block3 = ComplexModeBlock("K", k3, [0], np.array([[1.0 + 0j], [0.0 + 0j], [0.0 + 0j]]), 1, np.array([1.0 + 0j]), 0, {})
    modes3 = realify_star_complex_modes_to_supercell(block3, [k3], sc3, "cell_periodic_minus")
    assert len(modes3) == 2
    cos3 = modes3[0].reshape(sc3.supercell_crystal.nsites, 3)[:, 0]
    sin3 = modes3[1].reshape(sc3.supercell_crystal.nsites, 3)[:, 0]
    cos_by_shift3 = {
        tuple(int(x) for x in shift): float(cos3[i])
        for i, shift in enumerate(sc3.supercell_atom_to_primitive_lattice_shift)
    }
    sin_by_shift3 = {
        tuple(int(x) for x in shift): float(sin3[i])
        for i, shift in enumerate(sc3.supercell_atom_to_primitive_lattice_shift)
    }
    # K=(1/3,1/3,0): Re/Im pattern must follow exp(2πi k·L).
    assert abs(cos_by_shift3[(0, 0, 0)] - 1.0) < 1e-8
    assert abs(cos_by_shift3[(1, 0, 0)] + 0.5) < 1e-8
    assert abs(cos_by_shift3[(0, 1, 0)] + 0.5) < 1e-8
    assert abs(cos_by_shift3[(1, 2, 0)] - 1.0) < 1e-8
    assert abs(sin_by_shift3[(0, 0, 0)]) < 1e-8
    assert sin_by_shift3[(1, 0, 0)] > 0.9
    assert sin_by_shift3[(0, 1, 0)] > 0.9
    assert sin_by_shift3[(2, 0, 0)] < -0.9

    P_nd = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=int)
    reps = enumerate_integer_coset_representatives(P_nd)
    assert len(reps) == 2
    sc_nd = build_commensurate_supercell(primitive, P_nd, tol=1e-8)
    assert sc_nd.det == 2
    assert sc_nd.supercell_crystal.nsites == 2
