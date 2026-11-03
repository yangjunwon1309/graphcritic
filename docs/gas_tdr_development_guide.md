# GAS-style Temporal Distance Representation: development guide

**Status:** implementation plan; no TDR code is implemented by this document.  
**Scope:** explain the current `h=5` target and graph stitching, then specify the next implementation stages for a GAS-style temporal distance representation (TDR) in GraphCritic.

## 1. Current `h=5` target and the cap

### 1.1 What the graph stores

Let `G` be the set of graph nodes labelled successful. For node `u`, the graph builder stores

\[
C_G(u)=\min_{\pi:u\leadsto G}\sum_{e\in\pi}w_e,
\]

where `\pi` is a **directed** path to any goal node. Goal nodes have cost zero. A node with no directed path to a goal has `C_G(u)=+\infty`. With the current `weight_policy: lcb`, an edge has weight

\[
w_e=-\log(p^{\mathrm{LCB}}_e)+c\,\kappa_e,
\qquad c=-\log\gamma.
\]

The artifact keeps these graph costs in nats. **The graph cost itself is not clipped at `c_max_steps`.** The cap is applied only when a graph cost is converted into a critic target. See `graphcritic/weights.py`, `graphcritic/solve.py`, and `graphcritic/targets.py`.

### 1.2 How one valid `h=5` row is evaluated

For a chunk starting at transition row `i`, inspect its five transitions `i, i+1, ..., i+4`.

1. The chunk is valid if all five rows exist, they belong to one sampler segment, and none of the first four rows is terminal. A terminal on row `i+4` is allowed. Invalid rows get `valid=False` and a stored target of zero; the loss must ignore them with the validity mask.
2. If the current state `s_i` is already labelled successful, its remaining-cost target is zero.
3. Otherwise, scan the five next-state labels `s_{i+1}, ..., s_{i+5}`. If the first success is after `j` transitions, where `j` is in `{1,2,3,4,5}`, set `y_i=j`. The observed success in the chunk takes precedence over any graph estimate.
4. If no next state in the five-row chunk is successful, take the graph cost of the last next state, `s_{i+5}`. Convert nats to discounted time units, cap that remaining part, and add the five already-observed steps.

The code’s exact no-success branch is

\[
y_i=5+\min\left(\frac{C_G(\nu'_{i+4})}{c},\ c_{\max}^{\mathrm{steps}}\right),
\qquad c=-\log\gamma,
\]

where `\nu'_{i+4}` is the graph node assigned to the last transition’s next state. If `C_G=+\infty`, the minimum is `c_max_steps`. The maximum step-unit target is therefore `5 + c_max_steps`.

### 1.3 Values for the saved radius-0.85 artifact

The saved configuration has `gamma=0.99`, so

\[
c=-\log(0.99)\approx0.0100503\text{ nats/step}.
\]

Its `c_max_steps` is 1001, derived by the current pipeline from the longest segment length plus one. Therefore:

\[
C_{\max}=c\cdot1001\approx10.0604\text{ nats},
\qquad y_{\max}=h+c_{\max}^{\mathrm{steps}}=5+1001=1006\text{ steps}.
\]

Examples for a valid chunk with no success in the five transitions:

| Graph cost at endpoint | Remaining graph estimate | Target |
|---|---:|---:|
| `3` nats | `3 / 0.0100503 ≈ 298.5` steps | `≈303.5` steps |
| `30` nats | `≈2985` steps, clipped to `1001` | `1006` steps |
| `+∞` (no directed route to a goal) | clipped to `1001` | `1006` steps |

Thus a capped target means **“the remaining graph estimate is at least the configured ceiling, or the goal is unreachable in this graph.”** It does not mean the underlying graph cost equals the cap, nor that the environment’s true time-to-goal is exactly 1001 steps.

In `outputs/graphs/scene-play-singletask-task1-v0/radius-0.85/diagnostics.json`, `valid_frac=0.996` and `frac_at_cmax=0.9969819`. The latter is calculated over valid targets only: about 99.698% of valid `h=5` rows are at the maximum target 1006. It is **not** a percentage of nodes or all graph states. This severe target saturation remains a separate issue from whether TDR improves node geometry; it should be reported in every TDR comparison.

## 2. How current graph stitching and direction work

### 2.1 Radius merges state occurrences into nodes

The current node builder forms a state-occurrence table from all current states plus each segment’s final next state. It standardizes the configured raw state coordinates, then greedily groups occurrences whose distance to a cluster leader is at most `GraphConfig.radius`. With `radius=0.85`, that threshold is in the standardized raw-state Euclidean metric.

Goal occurrences and non-goal occurrences are clustered separately. An occurrence marked successful cannot be merged into a non-success node, even if their features are within the radius.

If an occurrence from trajectory A and one from trajectory B fall into the same cluster, both map to the same node ID. That shared node is the current implementation’s cross-trajectory stitching mechanism. **The radius merge itself does not add an edge.**

### 2.2 Observed transitions assign directed edges

For every transition row `i`, the implementation reads its assigned current and next nodes:

\[
u_i=\texttt{assign\_cur}[i],\qquad v_i=\texttt{assign\_next}[i].
\]

When `u_i != v_i`, it records the directed data edge

\[
u_i\longrightarrow v_i.
\]

Repeated observations of the same ordered pair aggregate into counts. Because nonterminal next-state assignment is tied to the following row’s current-state assignment, each episode preserves its time order. Edges are **not** made bidirectional automatically. A reverse edge exists only if a reverse transition was observed in the data (or a future, explicit bridge-edge feature adds one).

Stitching happens when a path enters a shared node using an observed incoming edge from one trajectory and leaves it using an observed outgoing edge from another. The solver may reverse the edge list internally to run multi-source Dijkstra from the goal set; it then returns costs for the original node-to-goal direction. That solver detail does not make the stored graph undirected.

## 3. Why evaluate GAS-style TDR here

GAS learns a mapping `\psi(s)` whose Euclidean distance approximates the optimal number of steps between states:

\[
d^*(s,g)\approx\|\psi(s)-\psi(g)\|_2.
\]

Its TDR is learned from offline state transitions with a goal-conditioned temporal-difference/expectile objective. GAS then uses this learned temporal geometry to cluster temporally similar states and construct a graph. The GAS paper also uses Temporal Efficiency (TE) filtering to exclude low-efficiency states. The state-based Scene-play recipe uses `encoder=not_used` (no image encoder) and sets `way_steps=48`; this is evidence for a learned state representation, not evidence that a generic pretrained visual encoder is appropriate for this QC task. See the [GAS paper](https://arxiv.org/abs/2506.07744), [official Scene-play recipe](https://github.com/qortmdgh4141/GAS/blob/main/README.md#L314-L318), and GAS [node](https://github.com/qortmdgh4141/GAS/blob/main/K_utils/keynodes_utils.py) and [graph](https://github.com/qortmdgh4141/GAS/blob/main/K_utils/keygraph_utils.py) implementations.

For GraphCritic, a TDR metric is promising because the target is time-to-goal, whereas standardized raw-state Euclidean distance can group states that look numerically close but differ in how many control steps separate them. It is a hypothesis to evaluate, not a guaranteed improvement. In particular:

- The current `.85` radius is a raw standardized-state threshold. It cannot be carried over to TDR space.
- GAS’s Scene-play uses `way_steps=48`, but that is task- and representation-specific. It should not be copied as a GraphCritic radius.
- The paper’s TDR metric is symmetric Euclidean distance, while controllability and the observed transition graph can be directed. Use TDR first for node identity / geometry; retain direction on empirical edges.
- The current QC benchmark ID is `scene-play-singletask-task1-v0`; GAS’s published recipe names `scene-play-v0`. Check state field order, wrappers, task labels, and preprocessing before attempting checkpoint reuse. Do not assume a checkpoint is compatible just because the task names look similar.
- GAS adds distance-based edges and repairs strong connectivity with synthetic bidirectional links. Those links are not observed transitions. GraphCritic’s LCB weights interpret data support, so importing synthetic edges without explicit marking and a separate penalty would change the meaning of its costs.
- TDR may improve clustering while leaving the current 99.7% target-cap rate intact. Evaluate geometry, graph costs, and target saturation separately.

## 4. Proposed design

### 4.1 Preserve the existing graph semantics

The first implementation should change only the metric used to assign state occurrences to nodes:

1. Compute a deterministic TDR vector `z=\psi(s)` for each state in the current state table.
2. Cluster occurrences using distance in TDR space, while retaining the current separate goal/non-goal grouping.
3. Keep `NodeSet.rep_feat` as the original raw observation. It is currently validated against `TransitionSet` and acts as the reproducible representative, so replacing it with a latent vector would break its schema meaning.
4. Build `EdgeSet` exactly as today from observed row transitions `assign_cur[i] -> assign_next[i]`; aggregate direction and counts exactly as today.
5. Keep graph weighting, shortest path, goal-distribution semantics, and target clipping unchanged for the first TDR ablation.

This gives TDR a well-defined role: it decides when two recorded states represent the same graph node; the transition table decides which directed actions-of-time are supported by data.

### 4.2 Feature-transform interface

`FeatureConfig.name` currently has no effect; `Standardizer` always performs identity-plus-coordinate-standardization. Add an explicit transform interface rather than adding a special case inside the clustering loop. Proposed contract:

```python
class StateFeatureTransform(Protocol):
    def transform(self, states: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def metadata(self) -> Mapping[str, JSONValue]: ...
```

Requirements:

- Input is `(M, raw_state_dim)` and output is finite `float32` `(M, latent_dim)`.
- The exact same transform and preprocessing is used for current and terminal next states.
- Transform is deterministic at inference; no dropout, random augmentation, or minibatch-dependent normalization.
- Transform output is cached once per state occurrence during graph construction.
- Raw representatives stay in `NodeSet.rep_feat`; latent vectors are clustering data only.
- Configuration records transform name, checkpoint path for user convenience, checkpoint SHA-256 for identity, checkpoint format/version, input dimension, latent dimension, and distance metric. The content hash must depend on immutable transform identity (especially checkpoint bytes/hash and preprocessing), not only a user-supplied name or mutable path.
- A checkpoint with a mismatched input dimension or unknown preprocessing fails early with a clear schema error.

Proposed configuration shape (not yet implemented):

```yaml
feature:
  name: gas_tdr
  checkpoint: checkpoints/scene_play_tdr/params_1000000.pkl
  checkpoint_sha256: <computed-at-load-or-build>
  preprocessing: qc_scene_play_v1
  metric: euclidean
  standardize: false
radius: <TDR-distance threshold, calibrated on validation>
```

Do not overload `feature.standardize` to mean both raw-coordinate scaling and latent normalization. Name each operation and its order explicitly.

### 4.3 Checkpoint and training route

First inspect whether the official GAS TDR checkpoint and inference code accept this exact QC observation vector. If compatible, write a narrow, pinned adapter that loads only the representation and its preprocessing. Record source repository commit, checkpoint SHA-256, and observation-field mapping. Do not import GAS graph construction into this path.

If no compatible checkpoint exists, train a TDR for this task’s state vectors using the GAS objective and dataset boundaries. The TDR goal `g` is a relabelled state used to learn pairwise temporal geometry; it is not restricted to states carrying the benchmark’s success label. Define

\[
V_\psi(s,g)=-\|\psi(s)-\psi(g)\|_2,
\qquad
\delta= -\mathbf{1}[s\ne g]+\gamma\bar V_\psi(s',g)-V_\psi(s,g),
\]

and minimize the asymmetric expectile loss

\[
\mathcal{L}_{\mathrm{TDR}}=\mathbb{E}[|\tau-\mathbf{1}[\delta<0]|\,\delta^2].
\]

Sample goals using same-trajectory future states and dataset-wide states, exclude the `s=g` case as required by the parameterization, and never draw a future transition across a QC segment boundary. Split validation by whole episode/segment groups before model selection. Save model weights plus observation/preprocessing metadata and a deterministic inference fixture.

Start with state input and the task-specific offline data. A generic image-semantic encoder is not the first choice for this state-vector benchmark. Do not add TE filtering in the first TDR version: filtering can remove rare success states and useful transitions, confounding the metric change with data selection. Add TE as a later, separate ablation with success coverage reported.

### 4.4 TDR clustering and threshold calibration

Add a named TDR clustering mode; do not silently reinterpret `radius` in existing configs. Use an explicit metric definition. GAS’s paper defines Euclidean norm, so the initial mode should use

\[
d_\psi(s_i,s_j)=\|\psi(s_i)-\psi(s_j)\|_2
\]

unless the chosen checkpoint’s official inference contract says otherwise. Define whether clustering is leader-radius assignment (current GraphCritic behavior) or nearest-centroid assignment (GAS paper description); implement and document one behavior, and include it in the graph hash. For the first comparable ablation, keep GraphCritic’s deterministic leader-radius clustering and change only the feature vectors.

Calibrate threshold in validation data without training a policy:

1. Measure TDR distances for held-out within-trajectory pairs at known temporal offsets (including short and long offsets), and inspect the distance-vs-step curve and rank correlation.
2. For candidate thresholds, report node count, state-to-node merge ratio, cluster diameter/size distribution, fraction of cross-trajectory merges, goal-node count, and success/non-success separation.
3. Report directed graph metrics and target metrics for each threshold: reachability, weak components, cost monotonicity/rank correlation, valid target fraction, and fraction at `h+c_max_steps`.
4. Choose a small fixed threshold set from this analysis before running QC policy training. Treat it as a predeclared graph ablation, not online benchmark tuning.

`way_steps=48`, GAS `te_threshold=0.99`, and the current raw-state `.85` must not be presented as transferable threshold values. The GAS reference numbers are useful starting context only after matching the task data and TDR implementation.

## 5. Staged implementation plan and acceptance criteria

### TDR-0 — Freeze the reference baseline

Record dataset artifact/hash, current graph config, code revision, and saved diagnostics for raw standardized state at radius `.85`. The present artifact reports 1,000,000 rows, 1,001,000 state occurrences, 643,058 nodes, 675,471 edges, 1,687 goal nodes, directed reachability `0.591504`, monotonicity `0.981400`, rank correlation `0.853955`, and `frac_at_cmax=0.996982` for `h=5`. Preserve these values as the comparison row. Do not use a newly rebuilt graph if its dataset or config hash differs.

### TDR-1 — Representation artifact and inference adapter

- Add a loader/adapter with strict input/output shape, finite-value, deterministic-output, and hash checks.
- Add a small saved fixture of raw observations and expected embeddings from the selected checkpoint.
- Add transform provenance to graph metadata and content hashing.
- Keep the current identity/standardized feature behavior byte-for-byte/config-for-config as a regression baseline.

**Accept when:** loading the chosen TDR artifact produces repeatable embeddings; incompatible task dimensions/preprocessing fail loudly; graph artifacts identify the exact model bytes and input mapping.

### TDR-2 — Graph node assignment by learned distance

- Refactor node construction to accept transformed clustering features separately from raw representative observations.
- Apply current goal/non-goal split and deterministic assignment order in TDR coordinates.
- Preserve all observed directed edge construction and aggregation unchanged.
- Give TDR threshold and distance units a distinct config field/name.

**Accept when:** every state occurrence maps to exactly one valid node; all existing `NodeSet` invariants hold; success labels never cross-merge; a transition’s direction is preserved; baseline identity config yields unchanged assignments and graph statistics.

### TDR-3 — Graph and target diagnostics

- Add TDR calibration diagnostics on held-out known-time pairs.
- Add cluster-size/diameter, cross-trajectory merge, and success-label coverage diagnostics.
- Build `h=1` and `h=5` targets for candidate thresholds using the unchanged target function.
- Report graph cost reachability and cap saturation side by side. Never summarize the cap rate without its denominator (valid targets).

**Accept when:** diagnostics include feature/checkpoint identity, metric and threshold units, dataset/hash, graph statistics, target validity and cap denominator, and no non-finite targets.

### TDR-4 — QC benchmark comparison

Compare at least:

1. Existing QC baseline without GraphCritic.
2. GraphCritic with raw standardized radius `.85`.
3. GraphCritic with TDR node clustering and the selected predeclared threshold(s).

Use identical dataset, algorithm, task, training budget, seeds, evaluation protocol, target horizon, and weight policy for the primary comparison. Since current LCB targets are heavily saturated, report that fact and include a geometry-isolating `weight_policy=time` ablation if the goal is to assess the node metric. Do not attribute changes to TDR if the graph weights or data filtering also changed. Save all graph/target artifacts with each run.

**Accept when:** the benchmark command can reproduce each run from saved config and artifacts; results include success rate, uncertainty across seeds, graph and target diagnostics, and all artifact hashes.

### TDR-5 — Optional GAS-style inferred edges (separate research step)

Only after TDR node geometry is validated, consider adding edges between nodes whose TDR distance is below an explicit temporal threshold. Keep them separate from observed transitions, mark them as bridge/inferred edges, specify their directionality, and assign bridge-specific uncertainty/penalties. Test directed and bidirectional variants separately. Never silently convert the current empirical graph to an undirected graph or treat inferred edge support as observed data.

**Accept when:** every inferred edge is identifiable in the artifact, its penalty is accounted for in costs, and the ablation compares against observed-edges-only construction under the same TDR nodes.

## 6. Planned code touch points

These are design targets, not existing APIs:

| Area | Expected change |
|---|---|
| `graphcritic/config.py` | Explicit transform/checkpoint/preprocessing/distance config and TDR threshold units |
| `graphcritic/featurize.py` | Transform protocol, identity standardizer compatibility, GAS TDR inference adapter, provenance |
| `graphcritic/nodes.py` | Cluster in transformed features while storing raw representative observations |
| `graphcritic/pipeline.py` | Fit/load transform once, transform the state table once, pass features into node construction, hash provenance |
| `graphcritic/types.py` / `graphcritic/io.py` | Persist and validate transform identity and any new metadata without changing raw `rep_feat` meaning |
| `experiments/build_graph.py` | CLI/config support for TDR checkpoint and diagnostics; keep old identity config usable |
| `graphcritic/diagnostics.py` | TDR calibration, cluster quality, and valid-target cap denominator |
| `configs/graph/` | Add an explicit TDR experiment config after threshold calibration |

The initial work item should be TDR-1 (checkpoint/data compatibility and feature-transform interface), followed by TDR-2. Do not begin by inserting GAS’s synthetic keygraph edges: that would combine two separate hypotheses and obscure whether the learned distance improved GraphCritic’s existing stitching mechanism.

## 7. Primary references

- Baek et al., [Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](https://arxiv.org/abs/2506.07744), especially TDR Eq. 3–6, TD-aware graph construction, and TE filtering.
- Official GAS [Scene-play recipe](https://github.com/qortmdgh4141/GAS/blob/main/README.md#L314-L318).
- Official GAS [key-node construction](https://github.com/qortmdgh4141/GAS/blob/main/K_utils/keynodes_utils.py) and [key-graph construction](https://github.com/qortmdgh4141/GAS/blob/main/K_utils/keygraph_utils.py).
- Current GraphCritic equations: [`docs/math.md`](math.md); implementation: `graphcritic/targets.py`, `graphcritic/nodes.py`, `graphcritic/edges.py`, `graphcritic/solve.py`.
