# -*- coding: utf-8 -*-
"""primitive-k complex mechanical representation 分解。

worker/POSCAR 写盘不在这里；本模块只把 little group / full star 上的 complex Bloch
表示分解成对称适配 block。外部标准表示命名层当前不参与主流程。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .models import ComplexModeBlock


def _hermitian(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.conj().T)


def _cluster_eigenvalues(w: np.ndarray, tol: float) -> List[List[int]]:
    if w.size == 0:
        return []
    scale = max(1.0, float(np.max(np.abs(w))))
    threshold = max(float(tol), float(tol) * scale)
    clusters: List[List[int]] = []
    cur = [0]
    w0 = float(w[0])
    for i in range(1, int(w.size)):
        if abs(float(w[i]) - w0) <= threshold:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
            w0 = float(w[i])
    clusters.append(cur)
    return clusters


def _decompose_complex_representation_blocks(
    Dk_ops: List[np.ndarray],
    seed: int,
    tol: float,
    high_symmetry_label: str = "K",
    k_vector: Optional[np.ndarray] = None,
    little_group_indices: Optional[List[int]] = None,
) -> List[ComplexModeBlock]:
    """内部线性代数 helper：用 complex Hermitian commutant 分解给定操作表示。"""
    if not Dk_ops:
        raise RuntimeError("Cannot decompose complex Bloch representation: little group operation list is empty.")
    n = int(Dk_ops[0].shape[0])
    rng = np.random.default_rng(int(seed))
    real_representation = all(float(np.linalg.norm(np.imag(D))) <= float(tol) for D in Dk_ops)
    if real_representation:
        # Γ、X、M、R 等 self-conjugate k 点常得到实表示；使用实对称 commutant
        # 可避免 complex eigenvector 的任意相位把同一实子空间重复成 Re/Im 两份。
        M_real = rng.standard_normal((n, n))
        M_real = 0.5 * (M_real + M_real.T)
        A_real = np.zeros((n, n), dtype=float)
        for D in Dk_ops:
            Dr = np.real(D)
            A_real += Dr @ M_real @ Dr.T
        A_real = 0.5 * (A_real + A_real.T) / float(len(Dk_ops))
        w, V_real = np.linalg.eigh(A_real)
        V = V_real.astype(complex)
    else:
        M0 = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        M = _hermitian(M0)
        A = np.zeros((n, n), dtype=complex)
        for D in Dk_ops:
            A += D @ M @ D.conj().T
        A = _hermitian(A / float(len(Dk_ops)))
        w, V = np.linalg.eigh(A)
    clusters = _cluster_eigenvalues(w, tol=float(tol))

    blocks: List[ComplexModeBlock] = []
    k_vec = np.zeros(3, dtype=float) if k_vector is None else np.asarray(k_vector, dtype=float).reshape(3)
    little = list(little_group_indices or list(range(len(Dk_ops))))
    for block_i, idxs in enumerate(clusters):
        B = V[:, idxs]
        character = np.array([np.trace(B.conj().T @ D @ B) for D in Dk_ops], dtype=complex)
        diagnostics = {
            "commutant_eigenvalue_min": float(np.min(w[idxs])),
            "commutant_eigenvalue_max": float(np.max(w[idxs])),
            "character_inner_norm_numeric": float(np.real(np.vdot(character, character)) / max(1, len(Dk_ops))),
            "real_representation": bool(real_representation),
            "label_status": "not_assigned",
            "label_note": "External standard representation naming is disabled in the main mode-only build path.",
        }
        blocks.append(ComplexModeBlock(
            high_symmetry_label=str(high_symmetry_label),
            k_vector=k_vec,
            little_group_indices=little,
            basis_complex=B,
            dimension_complex=int(B.shape[1]),
            character=character,
            block_index=int(block_i),
            diagnostics=diagnostics,
        ))
    return blocks


def decompose_star_induced_mechanical_representation(
    Dstar_ops: List[np.ndarray],
    seed: int,
    tol: float,
    high_symmetry_label: str,
    representative_k: np.ndarray,
    star_size: int,
) -> List[ComplexModeBlock]:
    """用完整空间群操作分解 full k-star induced mechanical representation。

    与 little-group 单 arm 分解不同，这里的表示空间已经包含所有 star arms，
    空间群操作可以在 arms 之间混合。返回的 basis_complex 维度是
    ``3 * N_primitive * star_size``。
    """
    blocks = _decompose_complex_representation_blocks(
        Dstar_ops,
        seed=seed,
        tol=tol,
        high_symmetry_label=high_symmetry_label,
        k_vector=np.asarray(representative_k, dtype=float),
        little_group_indices=list(range(len(Dstar_ops))),
    )
    out: List[ComplexModeBlock] = []
    for block in blocks:
        diag = dict(block.diagnostics)
        diag.update({
            "star_induced_representation": True,
            "star_size": int(star_size),
            "full_group_operation_count": int(len(Dstar_ops)),
        })
        out.append(ComplexModeBlock(
            high_symmetry_label=block.high_symmetry_label,
            k_vector=block.k_vector,
            little_group_indices=block.little_group_indices,
            basis_complex=block.basis_complex,
            dimension_complex=block.dimension_complex,
            character=block.character,
            block_index=block.block_index,
            diagnostics=diag,
        ))
    return out


def _selfcheck_kpoint_modes() -> None:
    D = [np.eye(3, dtype=complex)]
    blocks = _decompose_complex_representation_blocks(
        D,
        seed=123,
        tol=1e-8,
        high_symmetry_label="G",
        k_vector=np.zeros(3),
        little_group_indices=[0],
    )
    assert blocks
    assert blocks[0].diagnostics["label_status"] == "not_assigned"
