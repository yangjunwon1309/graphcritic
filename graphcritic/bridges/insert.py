"""Sequential graph updates for model-validated action-chunk bridges."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import sys
from typing import Callable, Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class IterativeBridgeGraph:
    """A G0 graph with exact decrease-only updates of costs and bridge counts."""

    n_nodes: int
    goals: np.ndarray
    base_src: np.ndarray
    base_dst: np.ndarray
    base_weight: np.ndarray
    cost: np.ndarray = field(init=False)
    successor: np.ndarray = field(init=False)
    bridge_count: np.ndarray = field(init=False)
    added_src: list[int] = field(default_factory=list)
    added_dst: list[int] = field(default_factory=list)
    added_weight: list[float] = field(default_factory=list)
    _reverse: csr_matrix = field(init=False, repr=False)
    _extra_predecessors: dict[int, list[tuple[int, float]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.goals = np.asarray(self.goals, dtype=np.int64)
        self.base_src = np.asarray(self.base_src, dtype=np.int64)
        self.base_dst = np.asarray(self.base_dst, dtype=np.int64)
        self.base_weight = np.asarray(self.base_weight, dtype=np.float64)
        if self.base_src.shape != self.base_dst.shape or self.base_src.shape != self.base_weight.shape:
            raise ValueError("base graph edge arrays must align")
        if np.any(self.base_weight <= 0) or not np.all(np.isfinite(self.base_weight)):
            raise ValueError("base graph weights must be finite and positive")
        self._reverse = csr_matrix(
            (self.base_weight, (self.base_dst, self.base_src)),
            shape=(self.n_nodes, self.n_nodes),
        )
        if self.goals.size:
            self.cost, pred, _ = dijkstra(
                self._reverse, directed=True, indices=self.goals,
                min_only=True, return_predecessors=True,
            )
            pred = np.asarray(pred, dtype=np.int64)
            self.successor = np.where(pred < 0, -1, pred).astype(np.int64)
        else:
            self.cost = np.full(self.n_nodes, np.inf, dtype=np.float64)
            self.successor = np.full(self.n_nodes, -1, dtype=np.int64)
        self.cost = np.asarray(self.cost, dtype=np.float64)
        self.cost[self.goals] = 0.0
        self.successor[self.goals] = -1
        self.bridge_count = np.zeros(self.n_nodes, dtype=np.int16)
        self.successor[~np.isfinite(self.cost)] = -1

    def optimistic(self, source: int, target: int, *, min_edge_cost: float, delta: float,
                   b_max: int) -> bool:
        s, t = int(source), int(target)
        if s == t or not np.isfinite(self.cost[t]):
            return False
        return bool(
            self.cost[s] - (min_edge_cost + self.cost[t]) >= delta
            and int(self.bridge_count[t]) + 1 <= b_max
        )

    def decrease_update(self, source: int, target: int, weight: float, *, b_max: int) -> bool:
        """Insert one directed edge and propagate all cost decreases upstream."""
        s, t, w = int(source), int(target), float(weight)
        if s == t or not np.isfinite(w) or w <= 0:
            raise ValueError("new bridge must be a non-self edge with positive finite cost")
        if not np.isfinite(self.cost[t]) or int(self.bridge_count[t]) + 1 > b_max:
            return False
        proposal = w + self.cost[t]
        proposal_b = int(self.bridge_count[t]) + 1
        if not (proposal < self.cost[s] - 1e-10 or
                (abs(proposal - self.cost[s]) <= 1e-10 and proposal_b < self.bridge_count[s])):
            return False

        self.added_src.append(s)
        self.added_dst.append(t)
        self.added_weight.append(w)
        self._extra_predecessors.setdefault(t, []).append((s, w))
        self.cost[s] = proposal
        self.bridge_count[s] = proposal_b
        self.successor[s] = t
        queue: list[tuple[float, int]] = [(proposal, s)]
        while queue:
            cur_cost, node = heapq.heappop(queue)
            if cur_cost > self.cost[node] + 1e-10:
                continue
            pred_start, pred_end = self._reverse.indptr[node:node + 2]
            base_pred = self._reverse.indices[pred_start:pred_end]
            base_w = self._reverse.data[pred_start:pred_end]
            extra = self._extra_predecessors.get(node, ())
            for pred, edge_w, is_bridge in (
                *((int(p), float(v), False) for p, v in zip(base_pred, base_w, strict=True)),
                *((int(p), float(v), True) for p, v in extra),
            ):
                new_cost = cur_cost + edge_w
                new_b = int(self.bridge_count[node]) + int(is_bridge)
                if new_b > b_max:
                    continue
                cheaper = new_cost < self.cost[pred] - 1e-10
                same_better = abs(new_cost - self.cost[pred]) <= 1e-10 and new_b < int(self.bridge_count[pred])
                if cheaper or same_better:
                    self.cost[pred] = new_cost
                    self.bridge_count[pred] = new_b
                    self.successor[pred] = int(node)
                    heapq.heappush(queue, (new_cost, int(pred)))
        return True

    def full_recompute(self, *, b_max: int = 32767) -> tuple[np.ndarray, np.ndarray]:
        """Recompute costs and minimum bridge counts for verification/tests."""
        cost, bridges, _ = self.full_recompute_paths(b_max=b_max)
        return cost, bridges

    def full_recompute_paths(self, *, b_max: int = 32767) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full constrained shortest-path solve, including successor pointers."""
        src = np.concatenate((self.base_src, np.asarray(self.added_src, dtype=np.int64)))
        dst = np.concatenate((self.base_dst, np.asarray(self.added_dst, dtype=np.int64)))
        weight = np.concatenate((self.base_weight, np.asarray(self.added_weight, dtype=np.float64)))
        # Lexicographic secondary objective: bridge count among equal cost paths.
        # Repeated Bellman relaxations are appropriate for tests and small graphs.
        cost = np.full(self.n_nodes, np.inf, dtype=np.float64)
        bridges = np.full(self.n_nodes, np.iinfo(np.int16).max, dtype=np.int16)
        successor = np.full(self.n_nodes, -1, dtype=np.int64)
        cost[self.goals] = 0.0
        bridges[self.goals] = 0
        reverse_order = np.lexsort((src, dst))
        changed = True
        while changed:
            changed = False
            for idx in reverse_order:
                s, t, w = int(src[idx]), int(dst[idx]), float(weight[idx])
                if not np.isfinite(cost[t]):
                    continue
                b = int(bridges[t]) + int(idx >= self.base_src.size)
                if b > b_max:
                    continue
                candidate = cost[t] + w
                if candidate < cost[s] - 1e-10 or (abs(candidate - cost[s]) <= 1e-10 and b < bridges[s]):
                    cost[s], bridges[s] = candidate, b
                    successor[s] = t
                    changed = True
        bridges[~np.isfinite(cost)] = 0
        return cost, bridges, successor


def insert_validated_bridges(
    graph: IterativeBridgeGraph,
    candidate_pairs: np.ndarray,
    validator: Callable[[np.ndarray], Iterable[object | None]],
    *,
    min_edge_cost: float,
    delta: float = 5.0,
    b_max: int = 2,
    max_candidates_per_round: int = 2048,
    lambda_cost: float = 0.0,
    progress: bool = False,
) -> list[dict[str, float | int]]:
    """Sequentially evaluate sorted candidates and update G0 path values."""
    pairs = np.asarray(candidate_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("candidate_pairs must have shape (N, 2)")
    accepted: list[dict[str, float | int]] = []
    round_index = 0
    while True:
        src, dst = pairs[:, 0], pairs[:, 1]
        target_finite = np.isfinite(graph.cost[dst])
        # Both endpoints can be unreachable initially; inf - inf is NaN and
        # is ineligible because the target must be finite. Keep that expected
        # case quiet while preserving the score ordering for valid candidates.
        with np.errstate(invalid="ignore"):
            score = graph.cost[src] - graph.cost[dst]
        eligible_mask = target_finite & (score >= min_edge_cost + delta)
        eligible_mask &= graph.bridge_count[dst] + 1 <= b_max
        eligible_ids = np.flatnonzero(eligible_mask)
        if eligible_ids.size > max_candidates_per_round:
            top = np.argpartition(score[eligible_ids], -max_candidates_per_round)[-max_candidates_per_round:]
            eligible_ids = eligible_ids[top]
        eligible_ids = eligible_ids[np.argsort(score[eligible_ids])[::-1]]
        batch = pairs[eligible_ids]
        if not len(batch):
            break
        added = 0
        for source, target in batch:
            s, t = int(source), int(target)
            if not graph.optimistic(s, t, min_edge_cost=min_edge_cost,
                                    delta=delta, b_max=b_max):
                continue
            plans = list(validator(np.asarray([[s, t]], dtype=np.int64)))
            if len(plans) != 1:
                raise ValueError("validator must return one plan-or-None per candidate")
            plan = plans[0]
            if plan is None:
                continue
            length = int(plan.length_chunks)
            h = int(plan.chunks.shape[1])
            weight = float(length * h + lambda_cost)
            improvement = float(graph.cost[s] - (weight + graph.cost[t]))
            if improvement + 1e-10 < delta:
                continue
            if graph.decrease_update(s, t, weight, b_max=b_max):
                accepted.append({
                    "order": len(accepted), "round": round_index,
                    "source": s, "target": t,
                    "length_chunks": length, "weight_steps": weight,
                    "improvement_steps": improvement,
                    "endpoint_error": float(plan.endpoint_error),
                    "sample_index": int(plan.sample_index),
                })
                added += 1
        if added == 0:
            break
        if progress:
            print(
                f"sequential bridge round {round_index}: added={added}, "
                f"total={len(accepted)}",
                file=sys.stderr, flush=True,
            )
        round_index += 1
    return accepted


def insert_validated_bridges_bulk(
    graph: IterativeBridgeGraph,
    candidate_pairs: np.ndarray,
    validator: Callable[[np.ndarray], Iterable[object | None]],
    *,
    min_edge_cost: float,
    delta: float = 5.0,
    b_max: int = 2,
    max_candidates: int = 2048,
    lambda_cost: float = 0.0,
    progress: bool = False,
) -> list[dict[str, float | int]]:
    """Comparison ablation: validate one G0-ranked batch, then add at once."""
    pairs = np.asarray(candidate_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("candidate_pairs must have shape (N, 2)")
    src, dst = pairs[:, 0], pairs[:, 1]
    with np.errstate(invalid="ignore"):
        score = graph.cost[src] - graph.cost[dst]
    mask = np.isfinite(graph.cost[dst]) & (score >= min_edge_cost + delta)
    mask &= graph.bridge_count[dst] + 1 <= b_max
    ids = np.flatnonzero(mask)
    if ids.size > max_candidates:
        ids = ids[np.argpartition(score[ids], -max_candidates)[-max_candidates:]]
    ids = ids[np.argsort(score[ids])[::-1]]
    batch = pairs[ids]
    if not len(batch):
        return []
    plans = list(validator(batch))
    if len(plans) != len(batch):
        raise ValueError("validator must return one plan-or-None per candidate")
    accepted = []
    for (s0, t0), plan in zip(batch, plans, strict=True):
        if plan is None:
            continue
        s, t = int(s0), int(t0)
        length, h = int(plan.length_chunks), int(plan.chunks.shape[1])
        weight = float(length * h + lambda_cost)
        improvement = float(graph.cost[s] - (weight + graph.cost[t]))
        if improvement + 1e-10 < delta:
            continue
        # All candidates use the same G0 C,b values: this is intentionally the
        # bulk ablation against sequential decrease-and-recheck insertion.
        graph.added_src.append(s)
        graph.added_dst.append(t)
        graph.added_weight.append(weight)
        graph._extra_predecessors.setdefault(t, []).append((s, weight))
        accepted.append({"order": len(accepted), "round": 0,
                         "source": s, "target": t,
                         "length_chunks": length, "weight_steps": weight,
                         "improvement_steps": improvement,
                         "endpoint_error": float(plan.endpoint_error),
                         "sample_index": int(plan.sample_index)})
    if accepted:
        graph.cost, graph.bridge_count, graph.successor = graph.full_recompute_paths(b_max=b_max)
    if progress:
        print(
            f"bulk bridge pass: candidates={len(batch)}, added={len(accepted)}",
            file=sys.stderr, flush=True,
        )
    return accepted

