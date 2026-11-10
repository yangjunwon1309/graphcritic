# Graph Critic v0.8.2b implementation and measured results

Date: 2026-11-10

This report records the revised Puzzle contact labels, event and motion quasimetric training, and the exact candidate-pair event audit. Artifacts are in the ignored `outputs/puzzle/p13-4x4-task2-100-generated/v08/` directory. Graph construction was gated on D2.5 and was not run after that gate failed.

## Implementation

- Contact/movement labels use raw Puzzle button-joint position coordinates. A button is marked moved when its five-frame displacement exceeds `0.05 × q99.9` for that coordinate; button-bit changes are included in the moved label.
- Contact intervals merge gaps shorter than five frames. Event constraints compare stationary windows immediately before and after each contact interval. The event timing statistic `T_I` is the first percentile of contact-start intervals.
- The label diagnostic reports the document's contact/board-change agreement, single-toggle purity, uncovered board changes, and `T_I` gates. Training refuses to start when those gates fail.
- The board audit supports exact cached candidate pairs and reports event-score separation by minimum press count.
- New object-model checkpoints carry the revised label version. The judged graph builder rejects checkpoints with the previous label version.

## D0: contact-label audit

The audit passed all four gates on P13 task 2:

| Metric | Measured |
|---|---:|
| Contact intervals / board changes | 30,119 / 30,074 |
| Relative count difference | 0.150% |
| Single-toggle pattern fraction | 1.000 |
| Uncovered board-change fraction | 0.253% (76 / 30,074) |
| Contact length p10 / p50 / p90 | 12 / 12 / 13 frames |
| `T_I` min / p1 / p5 | 26 / 28 / 29 frames |
| Event pre/post window pairs | 29,825 (99.02% of intervals) |
| Moved fraction under revised label | 37.58% |

The per-button movement thresholds were about 0.00143–0.00145 raw qpos units, derived from q99.9 five-frame displacements around 0.0285–0.0289.

## D2: 20k network training

Both networks were trained from scratch for 20,000 updates, with checkpoints and held-out reports at 10k and 20k. Neither emitted a collapse alarm; all reported per-layer dead-unit fractions were zero.

| Network | Held-out constraint pairs | Violation over cost at 20k | Violation over cost + ε | Other diagnostic |
|---|---:|---:|---:|---|
| Event (`cost=1`, `ε=0.25`) | 2,977 | 2.687% | 0.101% | Median constrained distance 0.682 |
| Motion (`cost=5 × moved chunks`, `ε=1.25`) | 99,287 | 43.249% | 0.119% | Spearman vs. moved-chunk count × 5: 0.8243 on 50k pairs |

For the motion network, the held-out distance median was 6.360; the reference moved-chunk count × 5 median was 10. The raw violation rate is high because it counts any exceedance of the cost, while the epsilon-tolerant rate is 0.119%.

The event network's sampled held-out event comparisons did not show useful score separation: at 20k, the median `eφ` was 1,727.1 for `n_p=0`, 1,733.7 for `n_p=1`, and 1,680.0 for `n_p≥2`. These sampled pairs are only a training diagnostic; the exact candidate audit below controls the graph gate.

## D2.5: exact candidate-pair audit

The event checkpoint was evaluated on all 6,111,200 pairs from the shared candidate cache. The required score ordering and false-positive gate both failed.

| Minimum press count | Candidate pairs | Median `eφ` |
|---|---:|---:|
| `n_p=0` | 1,381,599 | 0.0522 |
| `n_p=1` | 499,029 | 563.17 |
| `n_p≥2` | 4,230,572 | 2.9932 |
| `n_p=∞` | 0 | n/a |

The medians are not increasing with press count: `n_p=1` scores much higher than `n_p≥2`. At the specified `eφ≤1.5` threshold, the 2×2 table against `n_p≤1` was TP 1,084,033, FP 2,065,487, FN 796,595, TN 2,165,085. The false-positive rate among `n_p>1` candidates was **48.82%**, compared with the ≤1% gate.

Therefore D3 graph building and D4 retained-bridge auditing were not run. The motion model's promising rank correlation does not override the failed event-model gate, because D3 requires the event score to reject impossible board changes.

## Verification and artifacts

- Full test suite: `174 passed`, with three existing dependency/runtime warnings.
- `git diff --check`: passed.
- Label audit: `outputs/puzzle/p13-4x4-task2-100-generated/v08/object_labels_check_v082b.json`.
- Event/motion checkpoints: `.../v08/object2_events/step_020000` and `.../v08/object2_motion/step_020000`.
- Exact candidate audit: `outputs/puzzle/p13-4x4-task2-100-generated/v08/e_phi_check_v082b.json`.

Checkpoint metadata records `git_dirty: true`; the measurements were generated from the working tree changes described here and are not tied to a clean commit.
