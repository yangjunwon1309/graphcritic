"""Finite-state helpers for OGBench Lights Out puzzle analysis."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import SchemaError


_SHAPES = {9: (3, 3), 16: (4, 4), 20: (4, 5)}


def button_code(states: NDArray) -> int:
    values = np.asarray(states)
    if values.ndim != 1 or values.size not in _SHAPES or np.any((values != 0) & (values != 1)):
        raise SchemaError("button state must be a binary vector of width 9, 16, or 20")
    return int(np.dot(values.astype(np.int64), 1 << np.arange(values.size, dtype=np.int64)))


def decode_button_code(code: int, n_buttons: int) -> NDArray[np.int8]:
    if n_buttons not in _SHAPES or isinstance(code, bool) or not 0 <= int(code) < (1 << n_buttons):
        raise SchemaError("button code is outside the requested puzzle state space")
    return ((int(code) >> np.arange(n_buttons)) & 1).astype(np.int8)


def toggle_matrix(n_rows: int, n_cols: int) -> NDArray[np.uint8]:
    if n_rows * n_cols not in _SHAPES:
        raise SchemaError("Lights Out dimensions must describe 3x3, 4x4, or 4x5")
    n = n_rows * n_cols
    matrix = np.zeros((n, n), dtype=np.uint8)
    for pressed in range(n):
        row, col = divmod(pressed, n_cols)
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                matrix[rr * n_cols + cc, pressed] = 1
    return matrix


def solve_gf2(
    matrix: NDArray,
    rhs: NDArray,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """Return a random solution of ``A x = rhs`` over GF(2), if one exists."""
    a = np.asarray(matrix, dtype=np.uint8) & 1
    b = np.asarray(rhs, dtype=np.uint8).reshape(-1) & 1
    if a.ndim != 2 or a.shape[0] != b.size:
        raise SchemaError("GF(2) matrix and right-hand side are not aligned")
    rows, cols = a.shape
    aug = np.concatenate((a.copy(), b[:, None]), axis=1)
    pivots: list[tuple[int, int]] = []
    pivot_row = 0
    for col in range(cols):
        candidates = np.flatnonzero(aug[pivot_row:, col])
        if not candidates.size:
            continue
        selected = pivot_row + int(candidates[0])
        aug[[pivot_row, selected]] = aug[[selected, pivot_row]]
        for row in range(rows):
            if row != pivot_row and aug[row, col]:
                aug[row] ^= aug[pivot_row]
        pivots.append((pivot_row, col))
        pivot_row += 1
        if pivot_row == rows:
            break
    if np.any((aug[:, :cols].sum(axis=1) == 0) & (aug[:, cols] != 0)):
        raise SchemaError("Lights Out goal difference is not reachable from this board")
    pivot_columns = {col for _, col in pivots}
    free_columns = [col for col in range(cols) if col not in pivot_columns]
    free_values = (
        rng.integers(0, 2, size=len(free_columns), dtype=np.uint8)
        if rng is not None else np.zeros(len(free_columns), dtype=np.uint8)
    )
    solution = np.zeros(cols, dtype=np.uint8)
    solution[free_columns] = free_values
    for row, col in reversed(pivots):
        solution[col] = aug[row, cols] ^ (np.dot(aug[row, :cols], solution) & 1)
    if not np.array_equal((a @ solution) & 1, b):
        raise SchemaError("internal GF(2) elimination returned an invalid solution")
    return solution


def minimum_press_solution(matrix: NDArray, rhs: NDArray) -> NDArray[np.uint8]:
    """Find a minimum-Hamming-weight solution by enumerating the nullspace."""
    a = np.asarray(matrix, dtype=np.uint8) & 1
    b = np.asarray(rhs, dtype=np.uint8).reshape(-1) & 1
    rows, cols = a.shape
    aug = np.concatenate((a.copy(), b[:, None]), axis=1)
    pivots: list[tuple[int, int]] = []
    pivot_row = 0
    for col in range(cols):
        candidates = np.flatnonzero(aug[pivot_row:, col])
        if not candidates.size:
            continue
        selected = pivot_row + int(candidates[0])
        aug[[pivot_row, selected]] = aug[[selected, pivot_row]]
        for row in range(rows):
            if row != pivot_row and aug[row, col]:
                aug[row] ^= aug[pivot_row]
        pivots.append((pivot_row, col))
        pivot_row += 1
    if np.any((aug[:, :cols].sum(axis=1) == 0) & (aug[:, cols] != 0)):
        raise SchemaError("Lights Out goal difference is not reachable from this board")
    pivot_columns = {col for _, col in pivots}
    free_columns = [col for col in range(cols) if col not in pivot_columns]
    if len(free_columns) > 24:
        raise SchemaError("nullspace too large for exact minimum-press enumeration")
    best: NDArray[np.uint8] | None = None
    for free_bits in itertools.product((0, 1), repeat=len(free_columns)):
        solution = np.zeros(cols, dtype=np.uint8)
        if free_columns:
            solution[free_columns] = free_bits
        for row, col in reversed(pivots):
            solution[col] = aug[row, cols] ^ (np.dot(aug[row, :cols], solution) & 1)
        if best is None or int(solution.sum()) < int(best.sum()):
            best = solution
    if best is None:
        best = np.zeros(cols, dtype=np.uint8)
    return best


def apply_presses(board: NDArray, presses: NDArray, matrix: NDArray) -> NDArray[np.uint8]:
    state = np.asarray(board, dtype=np.uint8).reshape(-1) & 1
    actions = np.asarray(presses, dtype=np.uint8).reshape(-1) & 1
    toggle = np.asarray(matrix, dtype=np.uint8) & 1
    if toggle.shape != (state.size, actions.size):
        raise SchemaError("board, press vector, and toggle matrix dimensions do not agree")
    return ((state + toggle @ actions) & 1).astype(np.uint8)
