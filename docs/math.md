# Graph Critic v0.3 bridge formalization

This note describes the direct, closed-loop travel-time bridge schema. Graph construction uses only transitions, features, labels, modes, and optional simulator-state sidecars. It does not roll out a simulator policy. Simulator use is isolated to the later graph-tracking evaluation.

## State occurrences and position features

Each transition row `i` stores `(s_i, a_i, s'_i)`, episode/time labels, terminal status, success labels for both endpoints, and discrete modes. The state table contains all current observations followed by one final next-state occurrence for each terminal row. `graph_kind="state"` gives every occurrence its own vertex. Goal vertices are a distribution of success-labeled occurrences, not a single representative state.

Let `φ(s)` be the configured feature and let `d_step` be the median within-mode one-step Euclidean distance in that feature space. A step-unit radius `r` means `r × d_step` in feature units.

For OGBench Scene, `ogbench_scene_pos` uses the 40-value observation layout checked against the installed `SceneEnv.compute_observation()` implementation. It keeps eef xyz/yaw, gripper opening, cube xyz and `cos(4θ), sin(4θ)` computed from the sidecar quaternion, drawer position, and window position. It drops joint velocities, contact, cube quaternion/raw yaw, button observations, and drawer/window velocities. The 90-degree symmetric cube yaw encoding is invariant to the cube's four equivalent orientations. Each group is standardized per dimension and scaled by the inverse square root of its width. `robomimic_pos` applies the same rule to eef position plus sign-canonicalized quaternion, gripper width, and object xyz.

## Edge schema and costs

Temporal edge `(u,next(u))` has `κ=1` and weight `c=-log(γ)`.

A v0.3 bridge is a direct edge `(u,v)`: `m(u)=m(v)`, the feature distance satisfies the configured candidate radius, and same-episode pairs must satisfy `|t(v)-t(u)| > min_time_gap`. The destination can be any state occurrence, including a goal. The bridge's `anchor` is `v`, and its `κ` is the estimated travel time `τ(u,v)`. The graph does not attach copied actions or a simulated landing state to this edge.

The added switch charge is `cΛ₀`, so

\[
w(u,v)=c\bigl(\tau(u,v)+\Lambda_0\bigr),\qquad
C_G(u)=c\min_{\pi:u\leadsto\mathcal G}\left[
T_{\rm data}(\pi)+\sum_{b\in B(\pi)}(\tau_b+\Lambda_0)
\right].
\]

`tau_model="geometric"` uses the heuristic
\[
\tau(u,v)=\max\left(1,\left\lceil\rho\,d(\phi(u),\phi(v))/d_{\rm step}\right\rceil\right).
\]
`tau_model="learned"` uses the upper `q` quantile of a frame-gap classifier. The classifier estimates a distribution over `Δ=1,…,H` and `far`; a pair is rejected if `F_{uv}(H)<q`. The stored `κ` is `τ`, and the same `c(τ+Λ₀)` cost is used. Geometric time is a heuristic; learned time generalizes within-trajectory frame gaps to close cross-trajectory pairs (assumption A1″).

For a held-out near pair with observed gap `Δ`, upper-bound coverage is the event `Δ ≤ τ_q`, equivalently `F(Δ−1) < q`. Conformal calibration uses the episode-disjoint calibration split to select `q′` from these pre-gap CDF scores; test coverage is then measured directly as `mean(Δ ≤ τ_{q′})`. The far class is excluded from this coverage calculation.

Prior, data-calibrated, and simulator-verified open-loop costs remain available for comparison. They are legacy diagnostics/ablations; `landing_steps` is unused by the time policy. Beta lower-confidence penalties remain appropriate where real Bernoulli attempts are collected, such as the online BOOST interface, but do not define the v0.3 offline bridge cost.

## Graph solution and critic targets

Reverse multi-source Dijkstra computes the exact shortest-path cost on the finite graph. Unreachable vertices have infinite cost. For a valid horizon-`h` chunk with no success inside the chunk, the target is

\[
y_i=h+\min\left(C_G(\texttt{assign\_next}[i+h-1])/c,
 c_{\max}^{\rm steps}\right).
\]

Success inside a chunk still gives its first-success step count; an already-successful current state receives zero. The source codes and invalid-chunk behavior are unchanged from schema 0.3. Segment contraction preserves path costs exactly.

## Assumptions and simulator check

- **A1c (local closed-loop reachability):** for each bridge `(u,v)`, some policy reaches the tube around `v` within `τ_b` steps with probability at least `p_b`, where `cΛ₀ ≥ -log p_b`.
- **A2c (tube tracking):** inside each data-state tube, a policy can reach the next temporal-state tube in one step; a goal tube is within the success region.

Under these assumptions, following a shortest graph path costs no more than `C_G(u)/c` steps and succeeds with probability at least `exp(-C_G(u))`, giving `V*(u) ≥ exp(-C_G(u))`. For a learned model, A1c relies on frame-gap generalization to bridge candidates; calibration coverage is measured on held-out within-trajectory pairs and does not prove cross-trajectory reachability. The geometric model has no statistical coverage guarantee.

`GraphTrackingPolicy` and `experiments/eval_graph_policy.py` test a concrete closed-loop nearest-state policy after graph construction. They report `pi_g_success`, `pessimism_hold`, `time_ratio`, and per-bridge time ratios. This is diagnostic evidence for the assumptions, not a proof that every bridge is realizable. The graph build path must not invoke this policy or any simulator rollout.

## Public path representation

`extract_path(artifact, u)` follows the stored shortest-path successor chain and returns `PathStep(vertex, kind, tau)` entries. `kind=0` is temporal and `kind=1` is bridge; the final goal entry uses `kind=None, tau=0`. The edge-duration sum plus `Λ₀` for each bridge must equal `C_G(u)/c` for a time-policy graph.
