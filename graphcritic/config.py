"""Validated graph, bridge, feature, and target configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, Mapping

from graphcritic.types import SchemaError


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SchemaError(f"{name} must be a {qualifier} integer")


@dataclass(frozen=True)
class FeatureConfig:
    name: Literal[
        "identity", "select", "tdr", "ogbench_scene_pos", "robomimic_pos"
    ] = "identity"
    standardize: bool = True
    dims: tuple[int, ...] | None = None
    weights: tuple[float, ...] | None = None
    tdr_ckpt: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {
            "identity", "select", "tdr", "ogbench_scene_pos", "robomimic_pos"
        }:
            raise SchemaError("unknown feature.name")
        if not isinstance(self.standardize, bool):
            raise SchemaError("feature.standardize must be boolean")
        if self.name == "select" and not self.dims:
            raise SchemaError("feature.dims must be provided and non-empty for name='select'")
        if self.name in {"ogbench_scene_pos", "robomimic_pos"} and self.dims is not None:
            raise SchemaError(f"feature.dims is not supported for name={self.name!r}")
        if self.name in {"ogbench_scene_pos", "robomimic_pos"} and self.weights is not None:
            raise SchemaError("position features use fixed inverse-square-root group weights")
        if self.dims is not None:
            dims = tuple(self.dims)
            if not dims or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in dims):
                raise SchemaError("feature.dims must contain non-negative integer indices")
            if len(set(dims)) != len(dims):
                raise SchemaError("feature.dims must not contain duplicate indices")
            object.__setattr__(self, "dims", tuple(int(i) for i in dims))
        if self.weights is not None:
            weights = tuple(float(v) for v in self.weights)
            if not weights or any(not math.isfinite(v) or v <= 0 for v in weights):
                raise SchemaError("feature.weights must contain positive finite values")
            if self.dims is not None and len(weights) != len(self.dims):
                raise SchemaError("feature.weights must have the same length as feature.dims")
            object.__setattr__(self, "weights", weights)
        if self.name == "tdr" and (not isinstance(self.tdr_ckpt, str) or not self.tdr_ckpt):
            raise SchemaError("feature.tdr_ckpt is required when name='tdr'")


@dataclass(frozen=True)
class PessimismConfig:
    mode: Literal["union", "fixed"] = "union"
    delta: float = 0.05
    lam: float | None = None
    prior_data: tuple[float, float] = (1.0, 1.0)
    prior_bridge: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.mode not in {"union", "fixed"}:
            raise SchemaError("pessimism.mode must be 'union' or 'fixed'")
        if not isinstance(self.delta, (int, float)) or not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise SchemaError("pessimism.delta must be between zero and one")
        if self.mode == "fixed" and (
            self.lam is None or not isinstance(self.lam, (int, float))
            or not math.isfinite(self.lam) or self.lam <= 0
        ):
            raise SchemaError("pessimism.lam must be positive when mode='fixed'")
        if self.lam is not None and (
            not isinstance(self.lam, (int, float)) or not math.isfinite(self.lam) or self.lam <= 0
        ):
            raise SchemaError("pessimism.lam must be positive when provided")
        for name, prior in (("prior_data", self.prior_data), ("prior_bridge", self.prior_bridge)):
            try:
                valid = len(prior) == 2 and all(
                    isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in prior
                )
            except TypeError:
                valid = False
            if not valid:
                raise SchemaError(f"pessimism.{name} must contain two positive Beta parameters")
            object.__setattr__(self, name, tuple(float(v) for v in prior))


@dataclass(frozen=True)
class BridgeCostConfig:
    policy: Literal["time", "prior", "data_calibrated", "sim_verified"] = "time"
    tau_model: Literal["learned", "geometric"] = "learned"
    tau_ckpt: str | None = None
    tau_quantile: float = 0.9
    candidate_k: int = 32
    candidate_radius: float = 8.0
    rho: float = 1.5
    tau_max_steps: int = 48
    lambda0_steps: float = 5.0
    lambda1_steps: float = 10.0
    n_bins: int = 5
    action_tol: float = 0.5
    land_tol: float = 1.0
    delta: float = 0.05
    prior: tuple[float, float] = (1.0, 1.0)
    lambda_cap_steps: float = 200.0
    bridge_success_rho: float | None = None
    sim_budget: int = 2_000_000
    min_evidence: int = 50

    def __post_init__(self) -> None:
        if self.policy not in {"time", "prior", "data_calibrated", "sim_verified"}:
            raise SchemaError("bridge.cost.policy must be time, prior, data_calibrated, or sim_verified")
        if self.tau_model not in {"learned", "geometric"}:
            raise SchemaError("bridge.cost.tau_model must be learned or geometric")
        if self.tau_ckpt is not None and (not isinstance(self.tau_ckpt, str) or not self.tau_ckpt):
            raise SchemaError("bridge.cost.tau_ckpt must be a non-empty path when provided")
        if not isinstance(self.tau_quantile, (int, float)) or not 0 < self.tau_quantile < 1:
            raise SchemaError("bridge.cost.tau_quantile must lie strictly between zero and one")
        if not isinstance(self.candidate_radius, (int, float)) or not math.isfinite(self.candidate_radius) or self.candidate_radius <= 0:
            raise SchemaError("bridge.cost.candidate_radius must be finite and positive")
        if not isinstance(self.rho, (int, float)) or not math.isfinite(self.rho) or self.rho <= 0:
            raise SchemaError("bridge.cost.rho must be finite and positive")
        _positive_int("bridge.cost.candidate_k", self.candidate_k)
        _positive_int("bridge.cost.tau_max_steps", self.tau_max_steps)
        for name in ("lambda0_steps", "lambda1_steps", "action_tol", "land_tol", "lambda_cap_steps"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise SchemaError(f"bridge.cost.{name} must be finite and non-negative")
        if self.bridge_success_rho is not None and (
            not isinstance(self.bridge_success_rho, (int, float))
            or not math.isfinite(self.bridge_success_rho)
            or not 0 < self.bridge_success_rho <= 1
        ):
            raise SchemaError("bridge.cost.bridge_success_rho must lie in (0, 1]")
        if not 0 < self.delta < 1:
            raise SchemaError("bridge.cost.delta must be between zero and one")
        _positive_int("bridge.cost.n_bins", self.n_bins)
        _positive_int("bridge.cost.sim_budget", self.sim_budget)
        _positive_int("bridge.cost.min_evidence", self.min_evidence, allow_zero=True)
        if len(self.prior) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.prior):
            raise SchemaError("bridge.cost.prior must contain two positive finite values")
        object.__setattr__(self, "prior", tuple(float(v) for v in self.prior))


@dataclass(frozen=True)
class BridgeConfig:
    enabled: bool = True
    k: int = 8
    radius: float = 2.0
    radius_unit: Literal["abs", "step"] = "step"
    landing_steps: int = 5
    min_time_gap: int = 20
    source_stride: int = 1
    same_mode: bool = True
    knn_backend: Literal["auto", "ckdtree", "faiss", "jax_gpu"] = "auto"
    strategy: Literal["feature", "passive_r", "passive_r_real", "passive_r_strict", "passive_rm", "passive_rm_real", "puzzle_r"] = "feature"
    state_schema: str | None = None
    passive_jitter_multiplier: float = 1.0
    passive_max_multiplier: float = 10.0
    kinematic_settle: int = 2
    real_speed_quantile: float = 0.95
    real_settle_steps: int = 2
    free_distance: float = 0.05
    r_tau_max_steps: int | None = None
    include_m_bridges: bool = False
    cost: BridgeCostConfig = field(default_factory=BridgeCostConfig)

    def __post_init__(self) -> None:
        _positive_int("bridge.k", self.k)
        _positive_int("bridge.landing_steps", self.landing_steps)
        _positive_int("bridge.min_time_gap", self.min_time_gap, allow_zero=True)
        _positive_int("bridge.source_stride", self.source_stride)
        if not isinstance(self.enabled, bool):
            raise SchemaError("bridge.enabled must be boolean")
        if not isinstance(self.radius, (int, float)) or not math.isfinite(self.radius) or self.radius <= 0:
            raise SchemaError("bridge.radius must be finite and positive")
        if self.radius_unit not in {"abs", "step"}:
            raise SchemaError("bridge.radius_unit must be 'abs' or 'step'")
        if self.knn_backend not in {"auto", "ckdtree", "faiss", "jax_gpu"}:
            raise SchemaError("bridge.knn_backend must be auto, ckdtree, faiss, or jax_gpu")
        if self.strategy not in {"feature", "passive_r", "passive_r_real", "passive_r_strict", "passive_rm", "passive_rm_real", "puzzle_r"}:
            raise SchemaError("bridge.strategy must be feature, passive_r, passive_r_real, passive_r_strict, passive_rm, passive_rm_real, or puzzle_r")
        if self.strategy in {"passive_r", "passive_r_real", "passive_r_strict", "passive_rm", "passive_rm_real", "puzzle_r"} and self.cost.policy != "time":
            raise SchemaError("passive R/M bridge strategies require bridge.cost.policy='time'")
        if self.strategy == "puzzle_r" and self.state_schema != "ogbench_puzzle":
            raise SchemaError("puzzle_r requires bridge.state_schema='ogbench_puzzle'")
        if self.state_schema is not None and (not isinstance(self.state_schema, str) or not self.state_schema):
            raise SchemaError("bridge.state_schema must be a non-empty name when provided")
        for name in ("passive_jitter_multiplier", "passive_max_multiplier"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise SchemaError(f"bridge.{name} must be finite and positive")
        _positive_int("bridge.kinematic_settle", self.kinematic_settle, allow_zero=True)
        _positive_int("bridge.real_settle_steps", self.real_settle_steps, allow_zero=True)
        if not isinstance(self.real_speed_quantile, (int, float)) or not 0 < self.real_speed_quantile < 1:
            raise SchemaError("bridge.real_speed_quantile must lie strictly between zero and one")
        if not isinstance(self.free_distance, (int, float)) or not math.isfinite(self.free_distance) or self.free_distance <= 0:
            raise SchemaError("bridge.free_distance must be finite and positive")
        if self.r_tau_max_steps is not None:
            _positive_int("bridge.r_tau_max_steps", self.r_tau_max_steps)
        if not isinstance(self.include_m_bridges, bool):
            raise SchemaError("bridge.include_m_bridges must be boolean")
        if self.include_m_bridges and self.strategy != "passive_r_strict":
            raise SchemaError("bridge.include_m_bridges requires strategy='passive_r_strict'")
        if self.include_m_bridges and (
            self.cost.tau_model != "learned" or not self.cost.tau_ckpt
        ):
            raise SchemaError("bridge.include_m_bridges requires a learned tau checkpoint")
        if not isinstance(self.same_mode, bool):
            raise SchemaError("bridge.same_mode must be boolean")
        if not isinstance(self.cost, BridgeCostConfig):
            raise SchemaError("bridge.cost must be BridgeCostConfig")


@dataclass(frozen=True)
class GraphConfig:
    gamma: float = 0.99
    graph_kind: Literal["state", "node"] = "state"
    radius: float = 1.0
    radius_unit: Literal["abs", "step"] = "step"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    mode_fn: str = "none"
    min_dwell: int = 3
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    # Legacy node-graph controls, retained for graph_kind='node' comparisons.
    compress: bool = True
    consistency: bool = False
    weight_policy: Literal["hop", "time", "lcb"] = "lcb"
    pessimism: PessimismConfig = field(default_factory=PessimismConfig)
    c_max_steps: float | None = None
    segment_max_len: int | None = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.gamma, (int, float)) or not math.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise SchemaError("gamma must be between zero and one")
        if self.graph_kind not in {"state", "node"}:
            raise SchemaError("graph_kind must be 'state' or 'node'")
        if not isinstance(self.radius, (int, float)) or not math.isfinite(self.radius) or self.radius <= 0:
            raise SchemaError("radius must be positive")
        if self.radius_unit not in {"abs", "step"}:
            raise SchemaError("radius_unit must be 'abs' or 'step'")
        if not isinstance(self.feature, FeatureConfig):
            raise SchemaError("feature must be a FeatureConfig")
        if not isinstance(self.bridge, BridgeConfig):
            raise SchemaError("bridge must be a BridgeConfig")
        if not isinstance(self.pessimism, PessimismConfig):
            raise SchemaError("pessimism must be a PessimismConfig")
        if not isinstance(self.mode_fn, str) or not self.mode_fn:
            raise SchemaError("mode_fn must be a non-empty string")
        _positive_int("min_dwell", self.min_dwell)
        if not isinstance(self.compress, bool) or not isinstance(self.consistency, bool):
            raise SchemaError("compress and consistency must be booleans")
        if self.consistency:
            raise SchemaError("consistency=True is not supported")
        if self.weight_policy not in {"hop", "time", "lcb"}:
            raise SchemaError("weight_policy must be hop, time, or lcb")
        if self.c_max_steps is not None and (
            not isinstance(self.c_max_steps, (int, float))
            or not math.isfinite(self.c_max_steps) or self.c_max_steps <= 0
        ):
            raise SchemaError("c_max_steps must be positive when provided")
        if self.segment_max_len is not None:
            _positive_int("segment_max_len", self.segment_max_len)

    def validate(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GraphConfig:
        if not isinstance(value, Mapping):
            raise SchemaError("GraphConfig data must be a mapping")
        allowed = {
            "gamma", "graph_kind", "radius", "radius_unit", "feature", "mode_fn", "min_dwell",
            "bridge", "compress", "consistency", "weight_policy", "pessimism", "c_max_steps",
            "segment_max_len",
        }
        unknown = set(value) - allowed
        if unknown:
            raise SchemaError(f"unknown GraphConfig fields: {sorted(unknown)}")
        raw = dict(value)
        try:
            f = raw.get("feature", {})
            if isinstance(f, Mapping):
                f = dict(f)
                if f.get("dims") is not None:
                    f["dims"] = tuple(f["dims"])
                if f.get("weights") is not None:
                    f["weights"] = tuple(f["weights"])
                raw["feature"] = FeatureConfig(**f)
            p = raw.get("pessimism", {})
            if isinstance(p, Mapping):
                p = dict(p)
                for key in ("prior_data", "prior_bridge"):
                    if key in p:
                        p[key] = tuple(p[key])
                raw["pessimism"] = PessimismConfig(**p)
            b = raw.get("bridge", {})
            if isinstance(b, Mapping):
                b = dict(b)
                c = b.get("cost", {})
                if isinstance(c, Mapping):
                    c = dict(c)
                    if "prior" in c:
                        c["prior"] = tuple(c["prior"])
                    b["cost"] = BridgeCostConfig(**c)
                raw["bridge"] = BridgeConfig(**b)
            return cls(**raw)
        except TypeError as exc:
            raise SchemaError(f"invalid GraphConfig fields: {exc}") from exc

    def to_yaml(self, path: str | Path | None = None) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if path is not None:
            Path(path).write_text(serialized, encoding="utf-8")
        return serialized

    @classmethod
    def from_yaml(cls, source: str | Path) -> GraphConfig:
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            candidate = Path(source) if "\n" not in source else None
            try:
                is_file = candidate is not None and len(source) < 240 and candidate.is_file()
            except (OSError, ValueError):
                is_file = False
            text = candidate.read_text(encoding="utf-8") if is_file else source
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise SchemaError("non-JSON YAML requires optional PyYAML") from exc
            try:
                value = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise SchemaError("invalid YAML configuration") from exc
        return cls.from_dict(value)


@dataclass(frozen=True)
class TargetConfig:
    h: int
    unit: Literal["steps", "nats"] = "steps"
    kind: Literal["cost", "regret", "discounted", "gamma_pow"] = "regret"
    regret_max: float = 50.0
    discount_gamma: float | None = None

    def __post_init__(self) -> None:
        _positive_int("h", self.h)
        if self.unit not in {"steps", "nats"}:
            raise SchemaError("unit must be 'steps' or 'nats'")
        if self.kind not in {"cost", "regret", "discounted", "gamma_pow"}:
            raise SchemaError("kind must be 'cost', 'regret', 'discounted', or 'gamma_pow'")
        if self.kind in {"discounted", "gamma_pow"}:
            if self.unit != "steps":
                raise SchemaError(f"{self.kind} targets are defined in steps units")
            if (not isinstance(self.discount_gamma, (int, float))
                    or isinstance(self.discount_gamma, bool)
                    or not math.isfinite(self.discount_gamma)
                    or not 0.0 < self.discount_gamma < 1.0):
                raise SchemaError(f"{self.kind} targets require discount_gamma in (0,1)")
        elif self.discount_gamma is not None:
            raise SchemaError("discount_gamma is only valid for discounted and gamma_pow targets")
        if (not isinstance(self.regret_max, (int, float)) or not math.isfinite(self.regret_max)
                or self.regret_max <= 0):
            raise SchemaError("regret_max must be finite and positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetConfig:
        if not isinstance(value, Mapping):
            raise SchemaError("TargetConfig data must be a mapping")
        unknown = set(value) - {"h", "unit", "kind", "regret_max", "discount_gamma"}
        if unknown:
            raise SchemaError(f"unknown TargetConfig fields: {sorted(unknown)}")
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise SchemaError(f"invalid TargetConfig fields: {exc}") from exc
