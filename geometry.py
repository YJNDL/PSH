# -*- coding: utf-8 -*-
"""基础周期性几何工具。

坐标约定：
- lattice 为 (3, 3) 矩阵，三行分别是三个晶格矢量；
- 分数坐标使用行向量；
- 笛卡尔坐标转换为 ``cart = frac @ lattice``。

本模块只依赖 numpy，供 POSCAR IO、对称性匹配和 trial 构型生成复用。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def as_float_array(x: ArrayLike) -> np.ndarray:
    """把输入转换为 float numpy array。

    该 helper 统一接受 list/tuple/np.ndarray 等 ArrayLike，避免各模块重复写
    ``np.asarray(..., dtype=float)``。
    """
    return np.asarray(x, dtype=float)


def lattice_volume(lattice: ArrayLike) -> float:
    """返回晶胞体积 ``abs(det(lattice))``。

    lattice 的三行是晶格矢量，分数坐标行向量按 ``cart = frac @ lattice`` 转换。
    """
    lat = as_float_array(lattice)
    if lat.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {lat.shape}")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice contains non-finite values.")
    return abs(float(np.linalg.det(lat)))


def validate_lattice(lattice: ArrayLike) -> np.ndarray:
    """校验并返回 float lattice。

    检查 shape、有限值和非零体积；lattice 的三行是晶格矢量。
    """
    lat = as_float_array(lattice)
    if lat.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {lat.shape}")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice contains non-finite values.")
    volume = abs(float(np.linalg.det(lat)))
    if np.isclose(volume, 0.0, atol=1e-12):
        raise ValueError(f"lattice volume is too close to zero: volume={volume:.3e}")
    return lat


def _validate_last_dim3(x: np.ndarray, name: str) -> None:
    if x.shape == (3,):
        return
    if x.ndim < 1 or x.shape[-1] != 3:
        raise ValueError(f"{name} must have last dimension 3, got shape={x.shape}")


def wrap_frac(x: ArrayLike) -> np.ndarray:
    """把分数坐标 wrap 到 ``[0, 1)``。

    分数坐标使用行向量，后续笛卡尔转换约定为 ``cart = frac @ lattice``。
    """
    return np.mod(as_float_array(x), 1.0)


def wrap_diff(d: ArrayLike) -> np.ndarray:
    """把分数坐标差 wrap 到最小像区间 ``[-0.5, 0.5)``。

    使用 ``(d + 0.5) % 1.0 - 0.5``，避免 ``np.rint()`` 在精确 0.5
    边界处的 tie-breaking 差异。
    """
    arr = as_float_array(d)
    return (arr + 0.5) % 1.0 - 0.5


def wrap_to_mhalf_half(x: ArrayLike) -> np.ndarray:
    """兼容性别名：等价于 ``wrap_diff(x)``。"""
    return wrap_diff(x)


def normalize_direction(d: ArrayLike, eps: float = 1e-12) -> list[float]:
    """归一化 OP 方向向量；零向量按原值返回。"""
    arr = as_float_array(d).reshape(-1)
    dn = float(np.linalg.norm(arr))
    if dn < eps:
        return [float(x) for x in arr]
    return [float(x) / dn for x in arr]


def frac_to_cart(frac: ArrayLike, lattice: ArrayLike) -> np.ndarray:
    """分数坐标转笛卡尔坐标，约定 ``cart = frac @ lattice``。"""
    frac_arr = as_float_array(frac)
    _validate_last_dim3(frac_arr, "frac")
    lat = validate_lattice(lattice)
    return frac_arr @ lat


def cart_to_frac(cart: ArrayLike, lattice: ArrayLike) -> np.ndarray:
    """笛卡尔坐标转分数坐标。

    使用 ``np.linalg.solve(lat.T, cart.T).T``，避免显式构造逆矩阵。
    """
    cart_arr = as_float_array(cart)
    _validate_last_dim3(cart_arr, "cart")
    lat = validate_lattice(lattice)
    original_shape = cart_arr.shape
    cart_2d = cart_arr.reshape(-1, 3)
    frac_2d = np.linalg.solve(lat.T, cart_2d.T).T
    return frac_2d.reshape(original_shape)


def pbc_frac_diff(frac_a: ArrayLike, frac_b: ArrayLike) -> np.ndarray:
    """返回 ``frac_a - frac_b`` 的周期性最小像分数坐标差。"""
    a = as_float_array(frac_a)
    b = as_float_array(frac_b)
    _validate_last_dim3(a, "frac_a")
    _validate_last_dim3(b, "frac_b")
    return wrap_diff(a - b)


def pbc_cart_diff(frac_a: ArrayLike, frac_b: ArrayLike, lattice: ArrayLike) -> np.ndarray:
    """返回 ``frac_a - frac_b`` 的周期性最小像笛卡尔坐标差。"""
    return frac_to_cart(pbc_frac_diff(frac_a, frac_b), lattice)


def pbc_distance(frac_a: ArrayLike, frac_b: ArrayLike, lattice: ArrayLike) -> np.ndarray:
    """返回周期性最小像距离 ``||pbc_cart_diff||``。"""
    return np.linalg.norm(pbc_cart_diff(frac_a, frac_b, lattice), axis=-1)


def pbc_distance_matrix(source_frac: ArrayLike, target_frac: ArrayLike, lattice: ArrayLike) -> np.ndarray:
    """构造 source 到 target 的周期性距离矩阵。

    输出形状为 ``(n_source, n_target)``；内部使用 target-source 的最小像差，
    与原子映射代价矩阵的方向约定一致。
    """
    src = as_float_array(source_frac)
    tgt = as_float_array(target_frac)
    if src.ndim == 1:
        src = src.reshape(1, 3)
    if tgt.ndim == 1:
        tgt = tgt.reshape(1, 3)
    _validate_last_dim3(src, "source_frac")
    _validate_last_dim3(tgt, "target_frac")
    lat = validate_lattice(lattice)
    df = pbc_frac_diff(tgt[None, :, :], src[:, None, :])
    return np.linalg.norm(df @ lat, axis=2)
