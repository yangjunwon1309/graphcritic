"""State feature transforms and occurrence-table construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import FeatureConfig
from graphcritic.types import SchemaError, SimStateSet, TransitionSet


class Featurizer(Protocol):
    def fit(self, trans: TransitionSet) -> "Featurizer": ...
    def transform(self, x: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray[np.float32]: ...
    def meta(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StateFeaturizer:
    """Fitted identity/select transform or deterministic TDR checkpoint adapter."""

    config: FeatureConfig
    mean: NDArray[np.float32]
    std: NDArray[np.float32]
    input_dim: int
    output_weights: NDArray[np.float32]
    model: Any = None
    checkpoint_sha256: str | None = None
    group_ranges: tuple[tuple[str, int, int], ...] = ()
    layout: dict[str, Any] | None = None

    @classmethod
    def fit(
        cls,
        trans: TransitionSet,
        cfg: FeatureConfig,
        *,
        sim_state: SimStateSet | None = None,
    ) -> "StateFeaturizer":
        trans.validate()
        x, _, _, _ = state_table(trans)
        p = x.shape[1]
        aux = _sim_state_aux(sim_state, x.shape[0]) if sim_state is not None else None
        group_ranges: tuple[tuple[str, int, int], ...] = ()
        layout: dict[str, Any] | None = None
        if cfg.name in {"ogbench_scene_pos", "robomimic_pos"}:
            raw_features, group_ranges, layout = _position_features(x, cfg.name, aux)
            fit_values = raw_features
        else:
            dims = np.arange(p, dtype=np.int64) if cfg.dims is None else np.asarray(cfg.dims, dtype=np.int64)
            if np.any(dims >= p):
                raise SchemaError(f"feature.dims contains an index >= raw observation dimension {p}")
            if cfg.name == "tdr" and cfg.dims is not None:
                raise SchemaError("feature.dims is only valid for identity/select transforms")
            fit_values = x
        if cfg.standardize:
            mean64 = np.mean(fit_values, axis=0, dtype=np.float64)
            std64 = np.maximum(np.std(fit_values, axis=0, dtype=np.float64), 1e-6)
            mean = mean64.astype(np.float32)
            std = std64.astype(np.float32)
        else:
            mean = np.zeros(fit_values.shape[1], dtype=np.float32)
            std = np.ones(fit_values.shape[1], dtype=np.float32)

        model = None
        digest = None
        if cfg.name == "tdr":
            assert cfg.tdr_ckpt is not None
            path = Path(cfg.tdr_ckpt).expanduser().resolve()
            if not path.is_file():
                raise SchemaError(f"TDR checkpoint does not exist: {path}")
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
            digest = sha.hexdigest()
            model = _load_tdr_model(path)

        if cfg.name in {"identity", "select"}:
            output_dim = len(dims)
        elif cfg.name in {"ogbench_scene_pos", "robomimic_pos"}:
            output_dim = fit_values.shape[1]
        else:
            output_dim = None
        if cfg.name == "tdr":
            probe = _run_tdr_model(model, ((x[: min(32, len(x))] - mean) / std).astype(np.float32))
            if probe.ndim != 2 or probe.shape[0] != min(32, len(x)) or probe.shape[1] == 0:
                raise SchemaError("TDR encoder must return a two-dimensional embedding matrix")
            output_dim = int(probe.shape[1])
        assert output_dim is not None
        weights = np.ones(output_dim, dtype=np.float32)
        if group_ranges:
            for _, start, end in group_ranges:
                weights[start:end] = 1.0 / np.sqrt(end - start)
        if cfg.weights is not None:
            if len(cfg.weights) != output_dim:
                raise SchemaError(
                    f"feature.weights has {len(cfg.weights)} entries, expected {output_dim}"
                )
            weights = np.asarray(cfg.weights, dtype=np.float32)
        return cls(cfg, mean, std, p, weights, model, digest, group_ranges, layout)

    def transform(
        self,
        x: NDArray,
        aux: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float32]:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim == 0 or values.shape[-1] != self.input_dim:
            raise SchemaError("features have an incompatible final dimension")
        if not np.all(np.isfinite(values)):
            raise SchemaError("features must be finite")
        if self.config.name in {"ogbench_scene_pos", "robomimic_pos"}:
            one_dimensional = values.ndim == 1
            matrix = values.reshape(1, -1) if one_dimensional else values
            feature_aux = _reshape_aux(aux, matrix.shape[0])
            raw_features, _, _ = _position_features(matrix, self.config.name, feature_aux)
            normalized = (raw_features - self.mean) / self.std
            result = normalized * self.output_weights
            if one_dimensional:
                result = result[0]
        else:
            normalized = (values - self.mean) / self.std
        if self.config.name in {"identity", "select"}:
            dims = np.arange(self.input_dim, dtype=np.int64) if self.config.dims is None else self.config.dims
            result = normalized[..., dims] * self.output_weights
        elif self.config.name == "tdr":
            result = _run_tdr_model(self.model, normalized)
            if result.ndim != values.ndim or result.shape[:-1] != values.shape[:-1]:
                raise SchemaError("TDR transform must preserve all leading dimensions")
            if result.shape[-1] == 0:
                raise SchemaError("TDR transform returned an empty embedding")
            if self.config.weights is not None:
                if self.output_weights.size != result.shape[-1]:
                    raise SchemaError("feature.weights length must match the TDR embedding dimension")
                result = result * self.output_weights
        if not np.all(np.isfinite(result)):
            raise SchemaError("feature transform produced non-finite values")
        return np.ascontiguousarray(result, dtype=np.float32)

    def as_stats(self) -> dict[str, NDArray[np.float32]]:
        return {"mean": self.mean.copy(), "std": self.std.copy()}

    def meta(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "input_dim": self.input_dim,
            "output_dim": int(self.output_weights.size),
            "stats_dim": int(self.mean.size),
            "standardize": self.config.standardize,
            "dims": None if self.config.dims is None else list(self.config.dims),
            "weights": None if self.config.weights is None else list(self.config.weights),
            "checkpoint": None if self.config.tdr_ckpt is None else str(Path(self.config.tdr_ckpt).expanduser()),
            "checkpoint_sha256": self.checkpoint_sha256,
            "metric": "euclidean",
            "groups": [
                {"name": name, "start": start, "end": end, "weight": float(self.output_weights[start])}
                for name, start, end in self.group_ranges
            ],
            "layout": self.layout,
        }


def _sim_state_aux(sim_state: SimStateSet, n_states: int) -> dict[str, NDArray]:
    sim_state.validate(n_states)
    result: dict[str, NDArray] = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        result["button_states"] = sim_state.button_states
    return result


def _reshape_aux(
    aux: Mapping[str, NDArray] | None, n_rows: int
) -> dict[str, NDArray] | None:
    if aux is None:
        return None
    result: dict[str, NDArray] = {}
    for name, value in aux.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != n_rows:
            raise SchemaError(f"feature auxiliary field {name!r} must align with observations")
        result[name] = arr
    return result


def _position_features(
    x: NDArray,
    name: str,
    aux: Mapping[str, NDArray] | None,
) -> tuple[NDArray[np.float32], tuple[tuple[str, int, int], ...], dict[str, Any] | None]:
    if name == "ogbench_scene_pos":
        if aux is None or "qpos" not in aux:
            raise SchemaError("ogbench_scene_pos requires aligned sim_state qpos")
        from envs.ogbench_layout import scene_position_features

        button_states = aux.get("button_states")
        n_buttons = int(np.asarray(button_states).shape[1]) if button_states is not None else 2
        values, groups, layout = scene_position_features(x, aux["qpos"], n_buttons=n_buttons)
        return values, groups, layout.dimensions()
    if name == "robomimic_pos":
        from envs.ogbench_layout import robomimic_position_features

        values, groups = robomimic_position_features(x)
        return values, groups, {
            "eef_position": [0, 3], "eef_quaternion_sign_canonical": [3, 7],
            "gripper_qpos": [7, 9], "object_position": [9, 12],
        }
    raise SchemaError(f"{name!r} is not a registered position feature")


# The v0.1 name remains as an API alias for callers that only need scaling.
Standardizer = StateFeaturizer


def _load_tdr_model(path: Path) -> Any:
    """Load a deterministic TorchScript/PyTorch module or a safe linear NPZ encoder."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "weight" not in archive:
                raise SchemaError("linear TDR NPZ must contain a 'weight' matrix")
            weight = np.asarray(archive["weight"], dtype=np.float32)
            bias = np.asarray(archive["bias"], dtype=np.float32) if "bias" in archive else None
        if weight.ndim != 2 or (bias is not None and bias.shape != (weight.shape[0],)):
            raise SchemaError("linear TDR checkpoint must use weight (D_out,D_in) and bias (D_out,)")
        return (weight, bias)
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise SchemaError("TDR .pt/.ts checkpoints require the optional torch dependency") from exc
    try:
        model = torch.jit.load(str(path), map_location="cpu")
    except Exception:
        try:
            model = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise SchemaError(
                "unsupported TDR checkpoint; provide TorchScript, a callable torch module, or a linear NPZ"
            ) from exc
    if not callable(model):
        raise SchemaError("loaded TDR checkpoint is not an inference callable")
    if hasattr(model, "eval"):
        model.eval()
    return model


def _run_tdr_model(model: Any, values: NDArray[np.float32]) -> NDArray[np.float32]:
    if isinstance(model, tuple):
        weight, bias = model
        if weight.shape[1] != values.shape[-1]:
            raise SchemaError("linear TDR checkpoint input dimension does not match observations")
        result = values @ weight.T
        if bias is not None:
            result += bias
        return np.asarray(result, dtype=np.float32)
    try:
        torch = importlib.import_module("torch")
        with torch.no_grad():
            result = model(torch.as_tensor(values, dtype=torch.float32))
        if isinstance(result, dict):
            result = next((result[k] for k in ("embedding", "phi", "z", "features") if k in result), None)
        elif isinstance(result, (tuple, list)):
            result = result[0]
        if result is None:
            raise SchemaError("TDR module must return an embedding tensor")
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        return np.asarray(result, dtype=np.float32)
    except SchemaError:
        raise
    except Exception as exc:
        raise SchemaError(f"TDR inference failed: {exc}") from exc


def state_table(
    trans: TransitionSet,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64]]:
    """Return state observations, row references, goal labels, and mode labels.

    State IDs ``0..N-1`` refer to current states. Each terminal next-state
    occurrence follows in terminal-row order at IDs ``N..N+n_episodes-1``.
    """
    trans.validate()
    n_rows = trans.n_rows
    final_rows = np.flatnonzero(trans.terminal).astype(np.int64, copy=False)
    x = np.concatenate((trans.feat, trans.next_feat[final_rows]), axis=0).astype(np.float32, copy=False)
    ref_cur = np.column_stack((np.arange(n_rows, dtype=np.int64), np.zeros(n_rows, dtype=np.int64)))
    ref_final = np.column_stack((final_rows, np.ones(final_rows.size, dtype=np.int64)))
    ref = np.concatenate((ref_cur, ref_final), axis=0)
    success = np.concatenate((trans.success_cur, trans.success[final_rows])).astype(np.bool_, copy=False)
    modes = np.concatenate((trans.mode, trans.mode_next[final_rows])).astype(np.int64, copy=False)
    return tuple(np.ascontiguousarray(value) for value in (x, ref, success, modes))  # type: ignore[return-value]
