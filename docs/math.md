# Graph Critic formalization

This note fixes the notation used by the v0.1 row schema and graph-cost construction. It states the exact finite-graph result used by the implementation; statistical calibration and critic approximation guarantees require separate assumptions and are not claimed here.

## Data and graph notation

Let a flat dataset contain rows `i = 0, ..., N-1`, each representing one environment step

\[
(s_i,a_i,s'_i,r_i,d_i),
\]

where `d_i` marks the last row of its episode and `z_i` is the binary success label for the next state `s'_i`. The adapter supplies a feature map `ψ`, with `x_i = ψ(s_i)` and `x'_i = ψ(s'_i)`. Rows belonging to one episode are contiguous, and their local times start at zero and increase by one.

The graph builder assigns each current and next state to nodes:

\[
\nu_i = \texttt{assign\_cur}[i], \qquad
\nu'_i = \texttt{assign\_next}[i].
\]

For every nonterminal row, `ν'_i = ν_{i+1}`. A node is a goal exactly when all represented state occurrences are success states. The finite directed edge set is `E`; edge `e=(u,v)` stores duration upper bound `κ_e ≥ 1`, positive count `n_e^+`, negative count `n_e^-`, and kind (data or bridge).

## Edge costs and graph value

For discount `γ ∈ (0,1)`, let `c = -log γ` be the time cost per step. Given Beta prior `(α_0, β_0)`, the posterior lower confidence bound at quantile `q` is

\[
\ell_e(q) = F^{-1}_{\mathrm{Beta}(\alpha_0+n_e^+,\,\beta_0+n_e^-)}(q).
\]

For `mode="union"`, the v0.1 quantile is `q = δ / |E|`; for `mode="fixed"`, it is `q = exp(-λ)`. The edge cost is selected by policy:

\[
w_e =
\begin{cases}
1, & \text{hop},\\
c\,\kappa_e, & \text{time},\\
-\log \ell_e(q) + c\,\kappa_e, & \text{lcb}.
\end{cases}
\]

The cost-to-go from node `u` to the set of goal nodes `G` is

\[
C_G(u) = \begin{cases}
0, & u \in G,\\
\min_{(u,v)\in E}\{w_{(u,v)} + C_G(v)\}, & u \notin G\text{ and a goal is reachable},\\
\infty, & \text{otherwise}.
\end{cases}
\]

In an offline-only graph with `β_0=1` and `n_e^-=0`, the Beta quantile has the closed form

\[
-\log \ell_e(q) = \frac{\log(1/q)}{\alpha_0+n_e^+}.
\]

The implementation evaluates the general quantile with `scipy.special.betaincinv`. Artifact `cost` is stored in nats; unreachable nodes remain `inf`. Clipping is applied only when forming targets.

## Chunk targets and validity

For a chunk starting at row `i` with horizon `h`, define `valid_i` iff (1) `i+h-1 < N`, (2) all rows `i,...,i+h-1` belong to the same episode, and (3) none of `i,...,i+h-2` is terminal. This matches QC's `sample_sequence` endpoint-validity mask. Let

\[
k_i = \min\{j \in \{0,\ldots,h-1\}: z_{i+j}=1\},
\]

when the set is nonempty. Let `C_max = c · c_max_steps`. The step-unit target is

\[
y_i =
\begin{cases}
0, & \neg valid_i,\\
k_i+1, & valid_i\text{ and }k_i\text{ exists},\\
h + \min\left(C_G(\nu'_{i+h-1})/c,\;c_{\max}^{\mathrm{steps}}\right),
& valid_i\text{ and no success occurs in the chunk}.
\end{cases}
\]

Invalid targets are zero-filled because multiplying a NaN target by a zero loss mask still yields NaN. For `unit="nats"`, multiply the step target by `c`. Horizon `h=1` gives the FQL target. A cost critic can be trained with squared error to `y`; the actor consumes `Q=-C_\theta`.

For valid rows, the BOOST-style weighted-SFT weight is defined in nats as

\[
\omega_i = \operatorname{clip}\left(
\exp\left[\beta\left(C_G(\nu_i)-y_i^{\mathrm{nats}}\right)\right],
0,\omega_{\max}\right),
\]

and is zero for invalid rows. The graph cost and target must be converted to the same unit before evaluating this expression.

## Finite-graph shortest-path result

**Theorem (reverse multi-source Dijkstra).** Let `G=(V,E)` be a finite directed graph with a nonempty goal set `G_goal` and strictly positive finite edge weights. Reverse every edge while preserving its weight, then run multi-source Dijkstra from all goal nodes. The resulting distance for each original node `u` equals `C_G(u)`, the minimum sum of edge weights over all original directed paths from `u` to any goal. Nodes with no such path have infinite cost.

**Proof.** Reversing an original path from `u` to a goal produces a path from that goal to `u` with the same edge-weight sum. Thus paths from the multi-source set to `u` in the reversed graph are in one-to-one, cost-preserving correspondence with original paths from `u` to a goal. Dijkstra returns the minimum path cost for nonnegative weights, so its result is exactly the stated minimum; an unreachable node has no corresponding path and receives infinite distance. ∎

**Corollary (deterministic time-cost graph).** If all edge success penalties are zero and each edge has weight `c κ_e`, then `C_G(u) = c d^*(u)`, where `d^*(u)` is the minimum total duration `Σ κ_e` to a goal. In particular, for unit-duration edges this is `c` times the directed BFS hop distance.
