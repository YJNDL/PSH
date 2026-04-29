# -*- coding: utf-8 -*-
"""POSCAR 读写。

运行目录、父结构加载和 tee 日志在 runtime_io.py 中处理；本模块只负责 POSCAR
格式解析和写出。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .geometry import cart_to_frac, lattice_volume
from .models import Crystal
from .periodic_table import SYMBOL_TO_Z, Z_TO_SYMBOL


def _is_all_int(tokens: List[str]) -> bool:
    if not tokens:
        return False
    for token in tokens:
        try:
            int(token)
        except ValueError:
            return False
    return True


def _parse_poscar_scale(scale_line: str, path: Path) -> float:
    tokens = scale_line.split()
    if not tokens:
        raise ValueError(f"Missing POSCAR scale factor on line 2 in {path}")
    if len(tokens) == 3:
        raise ValueError(
            f"Three POSCAR scale factors are not supported in {path} line 2. "
            "Use a single positive scale or a single negative target volume."
        )
    if len(tokens) != 1:
        raise ValueError(f"Invalid POSCAR scale factor line in {path} line 2: {scale_line!r}")
    try:
        return float(tokens[0])
    except ValueError as e:
        raise ValueError(f"Cannot parse POSCAR scale factor in {path} line 2: {scale_line!r}") from e


def _parse_three_floats(line: str, path: Path, line_number: int, context: str) -> List[float]:
    tokens = line.split()
    if len(tokens) < 3:
        raise ValueError(
            f"Expected at least 3 numeric values for {context} in {path} line {line_number}, "
            f"got {len(tokens)}: {line!r}"
        )
    try:
        return [float(tokens[0]), float(tokens[1]), float(tokens[2])]
    except ValueError as e:
        raise ValueError(f"Cannot parse numeric values for {context} in {path} line {line_number}: {line!r}") from e


def read_poscar(path: str | Path) -> Crystal:
    path = Path(path)
    txt = path.read_text(encoding="utf-8").splitlines()
    if len(txt) < 8:
        raise ValueError(f"POSCAR too short: {path}")

    scale = _parse_poscar_scale(txt[1], path)
    raw_lattice = np.array([
        _parse_three_floats(txt[i], path, i + 1, "lattice vector")
        for i in range(2, 5)
    ], float)
    if scale > 0.0:
        lattice_scale = float(scale)
    elif scale < 0.0:
        raw_volume = lattice_volume(raw_lattice)
        if raw_volume <= 0.0:
            raise ValueError(f"Cannot apply negative POSCAR scale in {path}: lattice volume is zero.")
        lattice_scale = float((abs(scale) / raw_volume) ** (1.0 / 3.0))
    else:
        raise ValueError(f"POSCAR scale factor must be non-zero in {path} line 2.")
    lattice = raw_lattice * lattice_scale

    line5 = txt[5].split()
    if _is_all_int(line5):
        raise ValueError(
            "Detected VASP4-style POSCAR (no element symbol line). "
            "This script requires VASP5 format with element symbols on line 6."
        )
    symbols = line5
    try:
        counts = [int(x) for x in txt[6].split()]
    except ValueError as e:
        raise ValueError(f"Cannot parse POSCAR element counts in {path} line 7: {txt[6]!r}") from e
    if len(symbols) != len(counts):
        raise ValueError(
            f"POSCAR symbols/counts length mismatch in {path}: "
            f"line 6 has {len(symbols)} symbols, line 7 has {len(counts)} counts."
        )
    if any(count <= 0 for count in counts):
        raise ValueError(f"POSCAR element counts must be positive in {path} line 7: {counts}")
    idx = 7

    if txt[idx].strip().lower().startswith("s"):
        idx += 1
        if idx >= len(txt):
            raise ValueError(f"POSCAR ended after Selective dynamics line in {path}")

    coord_type = txt[idx].strip().lower()
    direct = coord_type.startswith("d")
    cartesian = coord_type.startswith("c") or coord_type.startswith("k")
    if not (direct or cartesian):
        raise ValueError(f"Cannot parse coordinate type line: '{txt[idx]}' in {path}")
    idx += 1

    n_sites = sum(counts)
    pos_lines = txt[idx: idx + n_sites]
    if len(pos_lines) < n_sites:
        raise ValueError(f"Not enough coordinate lines in POSCAR: need {n_sites}, got {len(pos_lines)}")

    coords = np.array([
        _parse_three_floats(ln, path, idx + offset + 1, "atomic coordinate")
        for offset, ln in enumerate(pos_lines)
    ], float)
    if direct:
        frac = coords % 1.0
    else:
        coords = coords * lattice_scale
        frac = cart_to_frac(coords, lattice)
        frac = frac % 1.0

    numbers = []
    for sym, count in zip(symbols, counts):
        z = SYMBOL_TO_Z.get(sym, None)
        if z is None:
            raise ValueError(f"Unknown element symbol '{sym}' in POSCAR.")
        numbers.extend([z] * count)

    return Crystal(lattice=lattice, frac=frac, numbers=np.array(numbers, int), symbols=symbols)


def _symbols_for_write(crys: Crystal) -> List[str]:
    symbols = list(crys.symbols or [])
    if symbols:
        return symbols

    inferred: List[str] = []
    seen = set()
    for z_raw in np.array(crys.numbers, dtype=int):
        z = int(z_raw)
        if z in seen:
            continue
        sym = Z_TO_SYMBOL.get(z)
        if sym is None:
            raise ValueError(
                "Cannot write POSCAR because crys.symbols is empty and "
                f"atomic number Z={z} cannot be converted to an element symbol."
            )
        inferred.append(sym)
        seen.add(z)
    if not inferred:
        raise ValueError("Cannot write POSCAR because crys.symbols is empty and no atoms are present.")
    return inferred


def write_poscar(crys: Crystal, path: str | Path, comment: str = "generated by landau scan") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    numbers = np.array(crys.numbers, dtype=int)
    frac_all = np.array(crys.frac, dtype=float)
    if frac_all.shape[0] != numbers.shape[0]:
        raise ValueError(
            f"Cannot write POSCAR {path}: frac/numbers length mismatch "
            f"({frac_all.shape[0]} vs {numbers.shape[0]})."
        )

    symbols = _symbols_for_write(crys)
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"Cannot write POSCAR {path}: duplicate symbols in crys.symbols: {symbols}")

    z_by_sym = []
    for sym in symbols:
        z = SYMBOL_TO_Z.get(sym)
        if z is None:
            raise ValueError(f"Cannot write POSCAR {path}: unknown element symbol '{sym}'.")
        z_by_sym.append(int(z))

    missing_z = sorted(set(int(z) for z in numbers) - set(z_by_sym))
    if missing_z:
        raise ValueError(
            f"Cannot write POSCAR {path}: crys.symbols={symbols} do not cover atomic numbers {missing_z}."
        )

    counts = [int(np.sum(numbers == z)) for z in z_by_sym]
    for sym, count in zip(symbols, counts):
        if count <= 0:
            raise ValueError(
                f"Cannot write POSCAR {path}: symbol '{sym}' has count={count}. "
                "Check crys.symbols and crys.numbers."
            )

    lines = [comment, "1.0"]
    for v in crys.lattice:
        lines.append(f"{v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}")
    lines.append(" ".join(symbols))
    lines.append(" ".join(str(c) for c in counts))
    lines.append("Direct")
    # POSCAR 坐标必须和 header 的 symbols/counts 顺序一致，因此按元素组稳定写出。
    for z in z_by_sym:
        for atom_i in np.flatnonzero(numbers == z):
            frac = frac_all[int(atom_i)] % 1.0
            lines.append(f"{frac[0]:16.10f} {frac[1]:16.10f} {frac[2]:16.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
