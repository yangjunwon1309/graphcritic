"""Validated, NumPy-only schemas shared by graph building and target generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import warnings
from numbers import Integral
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from graphcritic.config import GraphConfig


SCHEMA_VERSION = "0.1"


class SchemaError(ValueError):
    """Raised when an in-memory object or persisted artifact violates its schema."""


def _array(value: Any, dtype: Any, name: str) -> NDArray[Any]:
    target_dtype = np.dtype(dtype)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"{name} cannot be converted to {target_dtype}") from exc
    if raw.dtype.hasobject:
        raise SchemaError(f"{name} must not have object dtype")
    if np.issubdtype(target_dtype, np.integer):
        if not (np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_)):
            raise SchemaError(f"{name} must contain numeric integer values")
        if np.issubdtype(raw.dtype, np.number):
            if not np.all(np.isfinite(raw)) or np.any(raw != np.floor(raw)):
                raise SchemaError(f"{name} must contain finite integer values")
        limits = np.iinfo(target_dtype)
        if raw.size and (np.min(raw) < limits.min or np.max(raw) > limits.max):
            raise SchemaError(f"{name} contains values outside the {target_dtype} range")
    elif np.issubdtype(target_dtype, np.bool_):
        if not (np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_)):
            raise SchemaError(f"{name} must contain boolean values")
        if raw.size and np.any((raw != 0) & (raw != 1)):
            raise SchemaError(f"{name} must contain only boolean values")
    elif np.issubdtype(target_dtype, np.floating) and not np.issubdtype(raw.dtype, np.number):
        raise SchemaError(f"{name} must contain numeric values")
    try:
        result = np.asarray(raw, dtype=target_dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"{name} cannot be converted to {target_dtype}") from exc
    return np.ascontiguousarray(result)


def _jsonable_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, Mapping):
        raise SchemaError("config must be a GraphConfig or mapping")
    try:
        # Round-trip through JSON to reject arrays, custom objects, and non-finite values.
        return json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SchemaError("config must contain only finite JSON-compatible values") from exc


def _content_hash(
    schema_version: str,
    config: Any,
    arrays: Mapping[str, NDArray[Any]],
) -> str:
    """Hash a schema/configuration and named arrays in a platform-stable order."""
    digest = hashlib.sha256()
    digest.update(str(schema_version).encode("utf-8"))
    digest.update(b"\0")
    canonical_config = json.dumps(
        _jsonable_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    digest.update(canonical_config.encode("utf-8"))
    digest.update(b"\0")
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise SchemaError(f"array {name!r} must not have object dtype")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class TransitionSet:
    """Flat dataset rows; each row represents one environment step."""

    feat: NDArray[np.float32]
    next_feat: NDArray[np.float32]
    episode_id: NDArray[np.int64]
    t: NDArray[np.int64]
    success: NDArray[np.bool_]
    terminal: NDArray[np.bool_]
    row_idx: NDArray[np.int64]

    def __post_init__(self) -> None:
        self.feat = _array(self.feat, np.float32, "feat")
        self.next_feat = _array(self.next_feat, np.float32, "next_feat")
        self.episode_id = _array(self.episode_id, np.int64, "episode_id")
        self.t = _array(self.t, np.int64, "t")
        self.success = _array(self.success, np.bool_, "success")
        self.terminal = _array(self.terminal, np.bool_, "terminal")
        self.row_idx = _array(self.row_idx, np.int64, "row_idx")
        self.validate()

    @property
    def n_rows(self) -> int:
        return int(self.feat.shape[0])

    def validate(self) -> None:
        if self.feat.ndim != 2 or self.next_feat.ndim != 2:
            raise SchemaError("feat and next_feat must have shape (N, p)")
        if self.feat.shape[1] == 0 or self.next_feat.shape != self.feat.shape:
            raise SchemaError("feat and next_feat must have the same non-empty feature dimension")
        n_rows = self.feat.shape[0]
        for name in ("episode_id", "t", "success", "terminal", "row_idx"):
            if getattr(self, name).shape != (n_rows,):
                raise SchemaError(f"{name} must have shape ({n_rows},)")
        if not np.all(np.isfinite(self.feat)) or not np.all(np.isfinite(self.next_feat)):
            raise SchemaError("feat and next_feat must contain only finite values")
        if np.any(self.t < 0) or np.any(self.row_idx < 0):
            raise SchemaError("t and row_idx must be non-negative")
        if np.unique(self.row_idx).size != n_rows:
            raise SchemaError("row_idx values must be unique")
        if n_rows == 0:
            return

        starts = np.r_[0, np.flatnonzero(self.episode_id[1:] != self.episode_id[:-1]) + 1]
        ends = np.r_[starts[1:], n_rows]
        block_ids = self.episode_id[starts]
        if np.unique(block_ids).size != block_ids.size:
            raise SchemaError("rows from an episode must be contiguous")
        for start, end in zip(starts, ends, strict=True):
            if not np.array_equal(self.t[start:end], np.arange(end - start, dtype=np.int64)):
                raise SchemaError("t must start at zero and increase by one within each episode")
            if not np.all(~self.terminal[start : end - 1]) or not self.terminal[end - 1]:
                raise SchemaError("terminal must be true only on each episode's final row")
        nonterminal_rows = np.flatnonzero(~self.terminal)
        if nonterminal_rows.size and np.any(
            np.any(self.next_feat[nonterminal_rows] != self.feat[nonterminal_rows + 1], axis=1)
        ):
            warnings.warn(
                "next_feat differs from the following row's feat; this is allowed by the schema",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class NodeSet:
    """State nodes with explicit per-row current and next-state assignments."""

    rep_feat: NDArray[np.float32]
    rep_ref: NDArray[np.int64]
    assign_cur: NDArray[np.int64]
    assign_next: NDArray[np.int64]
    is_goal: NDArray[np.bool_]

    def __post_init__(self) -> None:
        self.rep_feat = _array(self.rep_feat, np.float32, "rep_feat")
        self.rep_ref = _array(self.rep_ref, np.int64, "rep_ref")
        self.assign_cur = _array(self.assign_cur, np.int64, "assign_cur")
        self.assign_next = _array(self.assign_next, np.int64, "assign_next")
        self.is_goal = _array(self.is_goal, np.bool_, "is_goal")
        self.validate()

    @property
    def n_nodes(self) -> int:
        return int(self.rep_feat.shape[0])

    def validate(self, trans: TransitionSet | None = None) -> None:
        if self.rep_feat.ndim != 2 or self.rep_feat.shape[1] == 0:
            raise SchemaError("rep_feat must have shape (V, p) with p > 0")
        n_nodes = self.rep_feat.shape[0]
        if self.rep_ref.shape != (n_nodes, 2):
            raise SchemaError(f"rep_ref must have shape ({n_nodes}, 2)")
        if self.is_goal.shape != (n_nodes,):
            raise SchemaError(f"is_goal must have shape ({n_nodes},)")
        if self.assign_cur.ndim != 1 or self.assign_next.shape != self.assign_cur.shape:
            raise SchemaError("assign_cur and assign_next must be equally sized one-dimensional arrays")
        if not np.all(np.isfinite(self.rep_feat)):
            raise SchemaError("rep_feat must contain only finite values")
        if np.any(self.assign_cur < 0) or np.any(self.assign_cur >= n_nodes):
            raise SchemaError("assign_cur contains an out-of-range node index")
        if np.any(self.assign_next < 0) or np.any(self.assign_next >= n_nodes):
            raise SchemaError("assign_next contains an out-of-range node index")
        if np.any(self.rep_ref[:, 0] < -1) or np.any(self.rep_ref[:, 0] >= self.assign_cur.size):
            raise SchemaError("rep_ref row must be -1 or a valid transition row")
        if np.any(~np.isin(self.rep_ref[:, 1], (0, 1))):
            raise SchemaError("rep_ref role must be 0 (current) or 1 (next)")

        if trans is None:
            return
        trans.validate()
        if self.assign_cur.shape != (trans.n_rows,):
            raise SchemaError("node assignments must have one entry per transition row")
        if self.rep_feat.shape[1] != trans.feat.shape[1]:
            raise SchemaError("node and transition feature dimensions differ")

        for node, (row, role) in enumerate(self.rep_ref):
            if row == -1:
                continue  # External graph builders such as BOOST may not expose source rows.
            source = trans.feat[row] if role == 0 else trans.next_feat[row]
            if not np.array_equal(self.rep_feat[node], source):
                raise SchemaError(f"rep_feat[{node}] does not match its rep_ref source")
            source_assignment = self.assign_cur[row] if role == 0 else self.assign_next[row]
            if source_assignment != node:
                raise SchemaError(f"rep_ref[{node}] does not point to a state assigned to that node")

        nonterminal_rows = np.flatnonzero(~trans.terminal)
        if nonterminal_rows.size and np.any(
            self.assign_next[nonterminal_rows] != self.assign_cur[nonterminal_rows + 1]
        ):
            raise SchemaError("assign_next[i] must equal assign_cur[i + 1] on nonterminal rows")

        # A successful transition labels its next state as a goal; the same state
        # is also the current state of the next row when the episode continues.
        cur_success = np.zeros(trans.n_rows, dtype=np.bool_)
        same_episode_next = trans.episode_id[1:] == trans.episode_id[:-1]
        cur_success[1:] = trans.success[:-1] & same_episode_next
        if np.any(self.is_goal[self.assign_cur] != cur_success):
            raise SchemaError("goal-node membership for current states disagrees with success labels")
        if np.any(self.is_goal[self.assign_next] != trans.success):
            raise SchemaError("goal-node membership for next states disagrees with success labels")


@dataclass
class EdgeSet:
    """Unique directed data/bridge edges and their occurrence statistics."""

    src: NDArray[np.int64]
    dst: NDArray[np.int64]
    kappa: NDArray[np.int64]
    n_pos: NDArray[np.int64]
    n_neg: NDArray[np.int64]
    kind: NDArray[np.int8]

    def __post_init__(self) -> None:
        self.src = _array(self.src, np.int64, "src")
        self.dst = _array(self.dst, np.int64, "dst")
        self.kappa = _array(self.kappa, np.int64, "kappa")
        self.n_pos = _array(self.n_pos, np.int64, "n_pos")
        self.n_neg = _array(self.n_neg, np.int64, "n_neg")
        self.kind = _array(self.kind, np.int8, "kind")
        self.validate()

    @property
    def n_edges(self) -> int:
        return int(self.src.size)

    def validate(self, n_nodes: int | None = None) -> None:
        n_edges = self.src.size
        for name in ("dst", "kappa", "n_pos", "n_neg", "kind"):
            if getattr(self, name).shape != (n_edges,):
                raise SchemaError(f"{name} must have shape ({n_edges},)")
        if np.any(self.src == self.dst):
            raise SchemaError("self-edges are not allowed")
        if np.any(self.kappa < 1):
            raise SchemaError("kappa must be at least one step")
        if np.any(self.n_pos < 0) or np.any(self.n_neg < 0):
            raise SchemaError("edge occurrence counts must be non-negative")
        if np.any(~np.isin(self.kind, (0, 1))):
            raise SchemaError("kind must be 0 (data) or 1 (bridge)")
        if n_nodes is not None and (
            np.any(self.src < 0)
            or np.any(self.dst < 0)
            or np.any(self.src >= n_nodes)
            or np.any(self.dst >= n_nodes)
        ):
            raise SchemaError("edge endpoint is outside the node table")
        if n_edges > 1:
            order = np.lexsort((self.kind, self.dst, self.src))
            same_as_previous = (
                (self.src[order][1:] == self.src[order][:-1])
                & (self.dst[order][1:] == self.dst[order][:-1])
                & (self.kind[order][1:] == self.kind[order][:-1])
            )
            if np.any(same_as_previous):
                raise SchemaError("(src, dst, kind) edge keys must be unique")


@dataclass
class GraphArtifact:
    """A graph, its derived costs, configuration, and reproducibility metadata."""

    config: GraphConfig
    feature_stats: dict[str, NDArray[np.float32]]
    nodes: NodeSet
    edges: EdgeSet
    weights: NDArray[np.float64]
    cost: NDArray[np.float64]
    succ: NDArray[np.int64]
    meta: dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    content_hash: str = ""

    def __post_init__(self) -> None:
        from graphcritic.config import GraphConfig as GraphConfigType

        if isinstance(self.config, Mapping):
            self.config = GraphConfigType.from_dict(self.config)
        if not isinstance(self.config, GraphConfigType):
            raise SchemaError("config must be a GraphConfig")
        if not isinstance(self.feature_stats, Mapping):
            raise SchemaError("feature_stats must be a mapping")
        self.feature_stats = {
            name: _array(values, np.float32, f"feature_stats.{name}")
            for name, values in self.feature_stats.items()
        }
        self.weights = _array(self.weights, np.float64, "weights")
        self.cost = _array(self.cost, np.float64, "cost")
        self.succ = _array(self.succ, np.int64, "succ")
        if not isinstance(self.meta, Mapping):
            raise SchemaError("meta must be a mapping")
        self.meta = dict(self.meta)
        supplied_hash = self.content_hash
        self.content_hash = ""
        self.validate(check_hash=False)
        calculated_hash = self.compute_hash()
        if supplied_hash and supplied_hash != calculated_hash:
            raise SchemaError("GraphArtifact content_hash does not match its contents")
        self.content_hash = calculated_hash

    def _arrays_for_hash(self) -> dict[str, NDArray[Any]]:
        arrays: dict[str, NDArray[Any]] = {
            "assign_cur": self.nodes.assign_cur,
            "assign_next": self.nodes.assign_next,
            "cost": self.cost,
            "dst": self.edges.dst,
            "is_goal": self.nodes.is_goal,
            "kappa": self.edges.kappa,
            "kind": self.edges.kind,
            "n_neg": self.edges.n_neg,
            "n_pos": self.edges.n_pos,
            "rep_feat": self.nodes.rep_feat,
            "rep_ref": self.nodes.rep_ref,
            "src": self.edges.src,
            "succ": self.succ,
            "weights": self.weights,
        }
        arrays.update({f"feature_{name}": value for name, value in self.feature_stats.items()})
        return arrays

    def compute_hash(self) -> str:
        return _content_hash(self.schema_version, self.config, self._arrays_for_hash())

    def validate(
        self,
        trans: TransitionSet | None = None,
        *,
        check_hash: bool = True,
    ) -> None:
        from graphcritic.config import GraphConfig as GraphConfigType

        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"unsupported graph schema_version {self.schema_version!r}")
        if not isinstance(self.config, GraphConfigType):
            raise SchemaError("config must be a GraphConfig")
        self.config.validate()
        self.nodes.validate(trans)
        self.edges.validate(self.nodes.n_nodes)
        n_nodes = self.nodes.n_nodes
        if self.weights.shape != (self.edges.n_edges,):
            raise SchemaError("weights must have one value per edge")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0):
            raise SchemaError("weights must be finite and strictly positive")
        if self.cost.shape != (n_nodes,) or self.succ.shape != (n_nodes,):
            raise SchemaError("cost and succ must have one value per node")
        if np.any(np.isnan(self.cost)) or np.any(self.cost < 0):
            raise SchemaError("cost must be non-negative and may use inf for unreachable nodes")
        if np.any(self.succ < -1) or np.any(self.succ >= n_nodes):
            raise SchemaError("succ contains an out-of-range node index")
        if np.any(self.cost[self.nodes.is_goal] != 0):
            raise SchemaError("goal-node costs must be zero")
        unreachable_or_goal = np.isinf(self.cost) | self.nodes.is_goal
        if np.any(self.succ[unreachable_or_goal] != -1):
            raise SchemaError("goal and unreachable nodes must have succ = -1")
        reachable_non_goal = np.isfinite(self.cost) & ~self.nodes.is_goal
        if np.any(self.succ[reachable_non_goal] < 0):
            raise SchemaError("reachable non-goal nodes must have a successor")
        if np.any(reachable_non_goal):
            pair_dtype = np.dtype([("src", np.int64), ("dst", np.int64)])
            edge_pairs = np.empty(self.edges.n_edges, dtype=pair_dtype)
            edge_pairs["src"] = self.edges.src
            edge_pairs["dst"] = self.edges.dst
            edge_pairs.sort(order=("src", "dst"))
            src_nodes = np.flatnonzero(reachable_non_goal)
            successor_pairs = np.empty(src_nodes.size, dtype=pair_dtype)
            successor_pairs["src"] = src_nodes
            successor_pairs["dst"] = self.succ[src_nodes]
            positions = np.searchsorted(edge_pairs, successor_pairs)
            in_bounds = positions < edge_pairs.size
            matches = np.zeros(src_nodes.size, dtype=np.bool_)
            matches[in_bounds] = edge_pairs[positions[in_bounds]] == successor_pairs[in_bounds]
            if not np.all(matches):
                raise SchemaError("succ must point along an edge in the graph")

        p = self.nodes.rep_feat.shape[1]
        if set(self.feature_stats) != {"mean", "std"}:
            raise SchemaError("feature_stats must contain exactly mean and std")
        for name, values in self.feature_stats.items():
            if values.shape != (p,) or not np.all(np.isfinite(values)):
                raise SchemaError(f"feature_stats.{name} must be a finite vector of shape ({p},)")
        if np.any(self.feature_stats["std"] < 1e-6):
            raise SchemaError("feature_stats.std must be at least 1e-6")

        required_meta = {"dataset_id", "source", "core_version", "n_rows", "c_max_steps"}
        if not required_meta.issubset(self.meta):
            raise SchemaError(f"meta is missing required keys: {sorted(required_meta - self.meta.keys())}")
        if not isinstance(self.meta["dataset_id"], str) or not self.meta["dataset_id"]:
            raise SchemaError("meta.dataset_id must be a non-empty string")
        if not isinstance(self.meta["source"], str) or not isinstance(self.meta["core_version"], str):
            raise SchemaError("meta.source and meta.core_version must be strings")
        n_rows = self.meta["n_rows"]
        if isinstance(n_rows, bool) or not isinstance(n_rows, Integral) or n_rows < 0:
            raise SchemaError("meta.n_rows must be a non-negative integer")
        if n_rows != self.nodes.assign_cur.size:
            raise SchemaError("meta.n_rows must equal the number of transition rows")
        if trans is not None and n_rows != trans.n_rows:
            raise SchemaError("meta.n_rows does not match the supplied TransitionSet")
        c_max_steps = self.meta["c_max_steps"]
        if (
            isinstance(c_max_steps, bool)
            or not isinstance(c_max_steps, (int, float))
            or not np.isfinite(c_max_steps)
            or c_max_steps <= 0
        ):
            raise SchemaError("meta.c_max_steps must be finite and positive")
        try:
            json.dumps(self.meta, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SchemaError("meta must contain only finite JSON-compatible values") from exc
        if check_hash:
            if not self.content_hash:
                raise SchemaError("GraphArtifact content_hash is required")
            if self.content_hash != self.compute_hash():
                raise SchemaError("GraphArtifact content_hash is stale or invalid")


@dataclass
class TargetSet:
    """Per-row chunk targets plus validity and provenance information."""

    y: NDArray[np.float32]
    valid: NDArray[np.bool_]
    h: int
    unit: str
    graph_hash: str
    dataset_id: str

    def __post_init__(self) -> None:
        self.y = _array(self.y, np.float32, "y")
        self.valid = _array(self.valid, np.bool_, "valid")
        if isinstance(self.h, bool) or not isinstance(self.h, Integral):
            raise SchemaError("h must be an integer")
        self.h = int(self.h)
        self.validate()

    def validate(self, trans: TransitionSet | None = None) -> None:
        if self.y.ndim != 1 or self.valid.shape != self.y.shape:
            raise SchemaError("y and valid must be equally sized one-dimensional arrays")
        if not np.all(np.isfinite(self.y)) or np.any(self.y < 0):
            raise SchemaError("targets must be finite and non-negative")
        if np.any(self.y[~self.valid] != 0):
            raise SchemaError("invalid targets must be filled with zero")
        if self.h < 1:
            raise SchemaError("h must be at least one")
        if self.unit not in {"steps", "nats"}:
            raise SchemaError("unit must be 'steps' or 'nats'")
        if not isinstance(self.graph_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", self.graph_hash):
            raise SchemaError("graph_hash must be a lowercase SHA-256 hex digest")
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise SchemaError("graph_hash and dataset_id must be non-empty")
        if trans is not None and self.y.shape != (trans.n_rows,):
            raise SchemaError("targets must have one value per transition row")
