"""Frame-gap distribution classifier used only to propose bridge travel times."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import SchemaError


_JAX_FORWARD_CACHE: dict[int, Any] = {}


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    values = np.asarray(logits, dtype=np.float32)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=-1, keepdims=True)


@dataclass(frozen=True)
class FrameGapModel:
    """Inference-only MLP with matching NumPy and JAX implementations."""

    params: Mapping[str, NDArray[np.float32]]
    obs_mean: NDArray[np.float32]
    obs_std: NDArray[np.float32]
    horizon: int
    metadata: Mapping[str, Any]
    checkpoint_sha256: str | None = None

    @property
    def input_dim(self) -> int:
        return int(self.obs_mean.size)

    @property
    def pair_delta_indices(self) -> NDArray[np.int64]:
        raw = self.metadata.get("pair_delta_indices")
        if raw is None:
            return np.arange(self.input_dim, dtype=np.int64)
        indices = np.asarray(raw, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.input_dim):
            raise SchemaError("tau pair_delta_indices are outside the state input")
        return indices

    def prepare_pairs(self, obs_u: NDArray, obs_v: NDArray) -> NDArray[np.float32]:
        u = np.asarray(obs_u, dtype=np.float32)
        v = np.asarray(obs_v, dtype=np.float32)
        if u.ndim == 1:
            u = u[None, :]
        if v.ndim == 1:
            v = v[None, :]
        if u.shape != v.shape or u.ndim != 2 or u.shape[1] != self.input_dim:
            raise SchemaError("tau model pairs must be aligned (B, observation_dim) arrays")
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            raise SchemaError("tau model inputs must be finite")
        zu = (u - self.obs_mean) / self.obs_std
        zv = (v - self.obs_mean) / self.obs_std
        delta_indices = self.pair_delta_indices
        return np.ascontiguousarray(
            np.concatenate((zu, zv, (zv - zu)[:, delta_indices]), axis=1), dtype=np.float32
        )

    def logits_numpy(self, pair_features: NDArray) -> NDArray[np.float32]:
        x = np.asarray(pair_features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.params["w0"].shape[0]:
            raise SchemaError("tau pair feature matrix has the wrong dimension")
        for layer in range(3):
            x = x @ self.params[f"w{layer}"] + self.params[f"b{layer}"]
            mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
            variance = np.mean(np.square(x - mean), axis=1, keepdims=True, dtype=np.float32)
            x = (x - mean) / np.sqrt(variance + np.float32(1e-5))
            x = x * self.params[f"ln_scale{layer}"] + self.params[f"ln_bias{layer}"]
            x = np.maximum(x, 0.0)
        return np.asarray(x @ self.params["w3"] + self.params["b3"], dtype=np.float32)

    def logits_jax(self, pair_features: NDArray) -> NDArray[np.float32]:
        try:
            jax = importlib.import_module("jax")
            jnp = importlib.import_module("jax.numpy")
        except ImportError as exc:  # pragma: no cover - bench-only route
            raise RuntimeError("JAX inference requires the optional JAX dependency") from exc
        key = id(self)
        forward = _JAX_FORWARD_CACHE.get(key)
        if forward is None:
            params = {name: jnp.asarray(value) for name, value in self.params.items()}

            @jax.jit
            def forward_fn(x):
                for layer in range(3):
                    x = jnp.matmul(
                        x, params[f"w{layer}"], precision=jax.lax.Precision.HIGHEST
                    ) + params[f"b{layer}"]
                    mean = jnp.mean(x, axis=1, keepdims=True)
                    variance = jnp.mean(jnp.square(x - mean), axis=1, keepdims=True)
                    x = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
                    x = x * params[f"ln_scale{layer}"] + params[f"ln_bias{layer}"]
                    x = jax.nn.relu(x)
                return jnp.matmul(
                    x, params["w3"], precision=jax.lax.Precision.HIGHEST
                ) + params["b3"]

            forward = forward_fn
            _JAX_FORWARD_CACHE[key] = forward
        return np.asarray(forward(jnp.asarray(pair_features, dtype=jnp.float32)), dtype=np.float32)

    def predict_proba_pairs(
        self,
        obs_u: NDArray,
        obs_v: NDArray,
        *,
        backend: str = "auto",
    ) -> NDArray[np.float32]:
        pairs = self.prepare_pairs(obs_u, obs_v)
        selected = backend
        if selected == "auto":
            try:
                jax = importlib.import_module("jax")
                selected = "jax" if jax.devices("gpu") else "numpy"
            except Exception:
                selected = "numpy"
        if selected == "numpy":
            logits = self.logits_numpy(pairs)
        elif selected == "jax":
            logits = self.logits_jax(pairs)
        else:
            raise ValueError("tau inference backend must be auto, numpy, or jax")
        return _softmax(logits)

    def calibrated_quantile(self, nominal: float) -> float:
        key = f"{nominal:.6g}"
        table = self.metadata.get("calibrated_quantiles", {})
        return float(table.get(key, nominal)) if isinstance(table, Mapping) else float(nominal)

    def predict_tau(self, probabilities: NDArray, quantile: float) -> tuple[NDArray, NDArray]:
        """Return integer upper-quantile travel time and ``F(H)``."""
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != self.horizon + 1:
            raise SchemaError("tau probabilities must contain H time classes and one far class")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise SchemaError("tau probabilities must be finite and non-negative")
        q = self.calibrated_quantile(quantile)
        if not np.isfinite(q) or not 0 < q < 1:
            raise SchemaError("tau quantile must lie strictly between zero and one")
        cdf = np.cumsum(p[:, : self.horizon], axis=1)
        f_h = cdf[:, -1]
        reaches = cdf >= q
        has_quantile = np.any(reaches, axis=1)
        tau = np.argmax(reaches, axis=1).astype(np.int64) + 1
        tau[~has_quantile] = self.horizon + 1
        return tau, f_h.astype(np.float32)

    def save(self, path: str | Path) -> None:
        npz_path = Path(path).expanduser()
        if npz_path.suffix != ".npz":
            npz_path = npz_path.with_suffix(".npz")
        json_path = npz_path.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, NDArray] = {
            "obs_mean": np.asarray(self.obs_mean, dtype=np.float32),
            "obs_std": np.asarray(self.obs_std, dtype=np.float32),
        }
        arrays.update({key: np.asarray(value, dtype=np.float32) for key, value in self.params.items()})
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(
            json.dumps(dict(self.metadata), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrameGapModel":
        npz_path = Path(path).expanduser()
        if npz_path.suffix == ".json":
            npz_path = npz_path.with_suffix(".npz")
        json_path = npz_path.with_suffix(".json")
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key], dtype=np.float32) for key in archive.files}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SchemaError(f"cannot load frame-gap checkpoint pair at {npz_path}") from exc
        if metadata.get("model") != "frame_gap_mlp_v1":
            raise SchemaError("unsupported frame-gap checkpoint model")
        horizon = int(metadata.get("horizon", 0))
        required = {"obs_mean", "obs_std"} | {
            name for layer in range(4) for name in (f"w{layer}", f"b{layer}")
        } | {name for layer in range(3) for name in (f"ln_scale{layer}", f"ln_bias{layer}")}
        if set(arrays) != required or horizon < 1:
            raise SchemaError("frame-gap checkpoint arrays or horizon are incomplete")
        mean, std = arrays.pop("obs_mean"), arrays.pop("obs_std")
        if mean.ndim != 1 or std.shape != mean.shape or np.any(std <= 0):
            raise SchemaError("frame-gap observation statistics are invalid")
        raw_delta_indices = metadata.get("pair_delta_indices")
        if raw_delta_indices is None:
            pair_dim = 3 * mean.size
        else:
            indices = np.asarray(raw_delta_indices, dtype=np.int64)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= mean.size):
                raise SchemaError("frame-gap pair_delta_indices are invalid")
            pair_dim = 2 * mean.size + indices.size
        if arrays["w0"].shape[0] != pair_dim or arrays["w3"].shape[1] != horizon + 1:
            raise SchemaError("frame-gap checkpoint dimensions do not match metadata")
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise SchemaError("frame-gap checkpoint arrays must be finite")
        hidden = int(arrays["w0"].shape[1])
        expected_shapes = {
            "w0": (pair_dim, hidden), "b0": (hidden,),
            "w1": (hidden, hidden), "b1": (hidden,),
            "w2": (hidden, hidden), "b2": (hidden,),
            "w3": (hidden, horizon + 1), "b3": (horizon + 1,),
        }
        expected_shapes.update({
            name: (hidden,) for layer in range(3)
            for name in (f"ln_scale{layer}", f"ln_bias{layer}")
        })
        if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
            raise SchemaError("frame-gap checkpoint layer dimensions are inconsistent")
        if metadata.get("observation_dim", mean.size) != mean.size:
            raise SchemaError("frame-gap observation dimension does not match metadata")
        calibration = metadata.get("calibrated_quantiles", {})
        if not isinstance(calibration, Mapping) or any(
            not np.isfinite(value) or not 0 < float(value) < 1
            for value in calibration.values()
        ):
            raise SchemaError("frame-gap calibrated quantiles must lie strictly between zero and one")
        digest = hashlib.sha256()
        for file_path in (npz_path, json_path):
            digest.update(file_path.name.encode("utf-8"))
            digest.update(b"\0")
            with file_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
        return cls(arrays, mean, std, horizon, metadata, digest.hexdigest())

