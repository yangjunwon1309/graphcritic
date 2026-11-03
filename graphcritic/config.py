"""Validated graph and target configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, Mapping

from graphcritic.types import SchemaError


@dataclass(frozen=True)
class FeatureConfig:
    name: str = "identity"
    standardize: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SchemaError("feature.name must be a non-empty string")
        if not isinstance(self.standardize, bool):
            raise SchemaError("feature.standardize must be boolean")


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
            self.lam is None
            or not isinstance(self.lam, (int, float))
            or not math.isfinite(self.lam)
            or self.lam <= 0
        ):
            raise SchemaError("pessimism.lam must be positive when mode='fixed'")
        if self.lam is not None and (
            not isinstance(self.lam, (int, float)) or not math.isfinite(self.lam) or self.lam <= 0
        ):
            raise SchemaError("pessimism.lam must be positive when provided")
        for name, prior in (("prior_data", self.prior_data), ("prior_bridge", self.prior_bridge)):
            try:
                valid_prior = len(prior) == 2 and all(
                    isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                    for value in prior
                )
            except TypeError:
                valid_prior = False
            if not valid_prior:
                raise SchemaError(f"pessimism.{name} must contain two positive Beta parameters")
            object.__setattr__(self, name, tuple(float(value) for value in prior))


@dataclass(frozen=True)
class GraphConfig:
    gamma: float = 0.99
    radius: float = 1.0
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    compress: bool = True
    consistency: bool = False
    weight_policy: Literal["hop", "time", "lcb"] = "lcb"
    pessimism: PessimismConfig = field(default_factory=PessimismConfig)
    c_max_steps: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gamma, (int, float)) or not math.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise SchemaError("gamma must be between zero and one")
        if not isinstance(self.radius, (int, float)) or not math.isfinite(self.radius) or self.radius <= 0:
            raise SchemaError("radius must be positive")
        if not isinstance(self.feature, FeatureConfig):
            raise SchemaError("feature must be a FeatureConfig")
        if not isinstance(self.pessimism, PessimismConfig):
            raise SchemaError("pessimism must be a PessimismConfig")
        if not isinstance(self.compress, bool):
            raise SchemaError("compress must be boolean")
        if self.consistency:
            raise SchemaError("consistency=True is not supported by the current graph implementation")
        if self.weight_policy not in {"hop", "time", "lcb"}:
            raise SchemaError("weight_policy must be 'hop', 'time', or 'lcb'")
        if self.c_max_steps is not None and (
            not isinstance(self.c_max_steps, (int, float))
            or not math.isfinite(self.c_max_steps)
            or self.c_max_steps <= 0
        ):
            raise SchemaError("c_max_steps must be positive when provided")

    def validate(self) -> None:
        """Recheck invariants, including values that may have been deserialized."""
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GraphConfig:
        if not isinstance(value, Mapping):
            raise SchemaError("GraphConfig data must be a mapping")
        allowed = {
            "gamma", "radius", "feature", "compress", "consistency",
            "weight_policy", "pessimism", "c_max_steps",
        }
        unknown = set(value) - allowed
        if unknown:
            raise SchemaError(f"unknown GraphConfig fields: {sorted(unknown)}")
        raw = dict(value)
        feature = raw.get("feature", {})
        pessimism = raw.get("pessimism", {})
        try:
            if isinstance(feature, Mapping):
                raw["feature"] = FeatureConfig(**feature)
            if isinstance(pessimism, Mapping):
                converted = dict(pessimism)
                for key in ("prior_data", "prior_bridge"):
                    if key in converted:
                        converted[key] = tuple(converted[key])
                raw["pessimism"] = PessimismConfig(**converted)
            return cls(**raw)
        except TypeError as exc:
            raise SchemaError(f"invalid GraphConfig fields: {exc}") from exc

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize as JSON, which is also a valid YAML 1.2 document."""
        serialized = json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if path is not None:
            Path(path).write_text(serialized, encoding="utf-8")
        return serialized

    @classmethod
    def from_yaml(cls, source: str | Path) -> GraphConfig:
        """Load JSON/YAML text or a path; PyYAML is only needed for non-JSON YAML."""
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
                raise SchemaError(
                    "non-JSON YAML requires the optional PyYAML dependency"
                ) from exc
            try:
                value = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise SchemaError("invalid YAML configuration") from exc
        return cls.from_dict(value)


@dataclass(frozen=True)
class TargetConfig:
    h: int
    unit: Literal["steps", "nats"] = "steps"

    def __post_init__(self) -> None:
        if isinstance(self.h, bool) or not isinstance(self.h, Integral) or self.h < 1:
            raise SchemaError("h must be an integer of at least one")
        if self.unit not in {"steps", "nats"}:
            raise SchemaError("unit must be 'steps' or 'nats'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetConfig:
        if not isinstance(value, Mapping):
            raise SchemaError("TargetConfig data must be a mapping")
        unknown = set(value) - {"h", "unit"}
        if unknown:
            raise SchemaError(f"unknown TargetConfig fields: {sorted(unknown)}")
        return cls(**dict(value))
