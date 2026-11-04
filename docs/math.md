# Graph Critic v0.2 graph formalization

This note describes the schema-0.3 implementation of the v0.2 graph redesign. It separates exact properties of the finite graph from assumptions about real simulator behavior and statistical calibration.

## State occurrences and replay data

Each transition row `i` stores `(s_i, a_i, s'_i)`, a segment ID, local time, terminal status, success labels for both endpoints, and discrete modes `m_i` and `m'_i`. The state table contains the `N` current-state occurrences followed by one final next-state occurrence for each terminal row. For a nonterminal row, its next-state occurrence is the next row's current state. `graph_kind="state"` gives each occurrence its own vertex; no geometric clustering occurs.

Let `φ(s)` be the configured identity, selected, or TDR feature. The median within-mode one-step feature distance is `d_step`. Bridge radius is absolute or `radius × d_step`.

The graph has two edge types:

- A temporal edge follows one recorded transition, has `κ=1`, and cost `c=-log γ`.
- A bridge from source `u` to anchor `v` replays the anchor's recorded actions for `L` steps and lands at `v+L`. It requires matching modes, an eligible time gap, distance within radius, and a valid same-mode landing segment. It has `κ=L`, stores `dist=d(φ(u),φ(v))`, and records `anchor=v`.

The bridge points to the landing state, not the anchor. It represents borrowing the anchor's action sequence for an open-loop replay. The source and anchor may be nearby but are distinct occurrences.

## Bridge cost policies

For a prior bridge, the added switch cost is

\[
\lambda_b = c\left(\Lambda_0 + \Lambda_1(d_b/r)^2\right),
\qquad w_b = cL + \lambda_b.
\]

Temporal edges never receive a confidence penalty. This is essential: recorded deterministic transitions are treated as certain, while uncertainty is charged at the cross-trajectory switch.

For data calibration, actions are standardized per action dimension. A candidate pair `(u,v)` contributes evidence when its L-step action-window RMS distance is at most `action_tol`. It is a success when its observed L-step endpoint distance is at most `land_tol × d_step` and the endpoint modes agree. Bridge state distances are partitioned into quantile bins. For each bin `k`,

\[
\ell_k = F^{-1}_{\operatorname{Beta}(\alpha_0+n_k^+,\,\beta_0+n_k^-)}(\delta/K),
\qquad \lambda_b = -\log \ell_{k(b)}.
\]

Here `K` is the number of nonempty distance bins. The bound is on the bin's mean landing-success probability, not an individual bridge. Applying it to a particular bridge relies on the additional assumption that action-matched evidence represents bridges in that distance bin. Edges whose added cost exceeds `lambda_cap_steps` are dropped.

For a deterministic simulator, `sim_verified` replays bridges on current shortest paths. A successful replay receives zero added cost; a failure is removed. Unused, unverified bridges are dropped when the verification budget ends. This validates the bridge landing tube, but not the local temporal tracking assumption after landing.

## Cost-to-go and critic targets

For each state occurrence `u`, the graph cost is

\[
C_G(u) = \min_{\pi:u\leadsto\mathcal{G}}
\left[c\,T(\pi)+\sum_{b\in B(\pi)}\lambda_b\right],
\]

where `T(π)` is the sum of replayed environment steps, `B(π)` is the set of bridges on the path, and `𝒢` is the set of success-labeled state occurrences. Reverse multi-source Dijkstra computes all costs exactly for the chosen finite graph and edge weights. Unreachable states have infinite graph cost; the target builder clips only at `c_max_steps`.

For a valid chunk beginning at row `i` with horizon `h`, a success on transition `i+k` yields `k+1` steps. If no success occurs inside the chunk, the endpoint is `assign_next[i+h-1]` and

\[
y_i = h + \min\left(C_G(\texttt{assign\_next}[i+h-1])/c,\;c_{\max}^{\mathrm{steps}}\right).
\]

An already-successful current state has target zero. Invalid chunks are zero-filled and marked invalid. `TargetSet.source` records `0=chunk success`, `1=finite graph cost`, `2=cap or unreachable`, and `3=invalid`.

## What the guarantee does and does not say

The finite-graph Dijkstra result is exact by path reversal. A value lower bound for the real environment needs stronger assumptions:

- **A1, bridge landing:** executing bridge `b` from source `u` reaches the landing tube around `v+L` with probability at least `p_b`.
- **A2, local tracking:** from any state in that tube, a policy can replay the landing state's following temporal segment to its next decision-state tube in the same number of steps.

If every bridge penalty satisfies `λ_b ≥ -log p_b`, following a shortest graph path gives

\[
V^*(u) \ge \exp(-C_G(u)).
\]

`sim_verified` checks A1 for the tested bridge resets. It does not prove A2; sampled path replay and `path_exec_rate` measure how well the composed assumption works. Data-calibrated costs support the stated bound only under the distance-bin representativeness assumption and its confidence event. These assumptions are reported separately from the exact shortest-path computation.

## Exact temporal-chain contraction

A state is a decision state when it starts an episode, is a goal, changes mode, starts or ends a bridge, ends a temporal chain, or reaches the configured segment-length limit. Between decision states the only outgoing choice is the recorded temporal successor. Replacing each such chain by one edge with summed duration and summed weight preserves every path cost between decision states, so shortest-path costs on those states are unchanged.
