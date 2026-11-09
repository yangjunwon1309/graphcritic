# Graph Critic v0.8 checkpoint recovery runs

**Implementation code commit:** `d6a7f97fd7b425dfe201e81f35752f6fd763e13a`

This runbook implements v0.8 §2.1–2.2 (T4). It only provides commands; the user runs all tests and training. No simulator or QC run is part of these steps.

## Prepare a clean, pinned worktree

Use the worktree at the implementation commit so checkpoint provenance records `git_dirty=false`. The commands below assume the repository is `/home/url/graphcritic`, its benchmark environment is already installed, and the clean worktree is `/home/url/gc-run-v08`.

```bash
cd /home/url/graphcritic
git worktree add --detach /home/url/gc-run-v08 quasi_develop   # only if this worktree does not already exist
git -C /home/url/gc-run-v08 checkout --detach d6a7f97fd7b425dfe201e81f35752f6fd763e13a
git -C /home/url/gc-run-v08 status --short                    # should print nothing
cd /home/url/gc-run-v08

ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated
CK=$D/v08/quasimetric
COMMON="--transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz"
```

Outputs for the short checks go under `$D/v08/check/`. The previous P13 checkpoints are under `$CK/`. No standard-task checkpoint is available from the interrupted training run.

## 0. Unit tests

```bash
$PY -m pytest tests/test_quasimetric_v08.py -q
```

**Expected time:** a few minutes, depending on whether the GPU grid smoke test is enabled.

**Check:** all tests pass. Stop here and send the failure output if any test fails.

## A0. Inspect the existing P13 checkpoints

```bash
$PY experiments/analyze_quasimetric_ckpt.py \
  --ckpt $CK/step_050000 $CK/step_100000 $CK/step_150000 $COMMON
```

**Expected time:** a few minutes. This only loads existing checkpoints and computes held-out diagnostics. It writes the comparison table and distance summaries to stdout.

**Check:** read the `projector_1` dead-unit fraction at step 50k. If it is **greater than 0.95**, start A2 and A3 from scratch by omitting `--init-from $CK/step_050000` from those commands. Otherwise use the warm-start commands below. Record the encoder/projector variances, λ, held-out constraint rates, and `P(d≤5 | j=5)` for all three checkpoints.

## A1. Reproduce the old configuration with collapse monitoring

```bash
$PY experiments/train_quasimetric.py $COMMON --resume-from $CK/step_100000 \
  --activation relu --no-layernorm --grad-clip 0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 --out $D/v08/check/a1_reproduce
```

**Expected time:** about 12 minutes. Metrics go to `$D/v08/check/a1_reproduce/train_metrics.jsonl`; saved checkpoints and held-out reports go to the same directory. The monitor may stop early with exit code `3`, after saving the alarm checkpoint. That code means the configured collapse gate fired; it is not a missing-file or CLI error.

**Check:** use the summary command below to inspect 5k-step rows and the alarm step. This run reproduces the ReLU/Adam-dual setup and checks that F3 catches collapse.

## A2. GELU, LayerNorm, clipping, Adam dual

Use the warm-start command if A0 found the 50k projector dead-unit fraction at or below 0.95:

```bash
$PY experiments/train_quasimetric.py $COMMON --init-from $CK/step_050000 \
  --activation gelu --layernorm --grad-clip 1.0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 --out $D/v08/check/a2_fix123
```

If A0 found it above 0.95, run the same command **without** `--init-from $CK/step_050000`.

**Expected time:** about 12 minutes. F1–F3 are enabled. Warm start copies the old Dense weights, IQE α, and λ; LayerNorm starts at scale 1 and bias 0, and optimizer state is fresh. An alarm saves a checkpoint and exits with code `3`.

## A3. Add projected-SGD dual update

Use the warm-start command if A0 found the 50k projector dead-unit fraction at or below 0.95:

```bash
$PY experiments/train_quasimetric.py $COMMON --init-from $CK/step_050000 \
  --activation gelu --layernorm --grad-clip 1.0 --dual sgd --dual-lr 1.0 \
  --steps 20000 --log-every 1000 --save-every 10000 --out $D/v08/check/a3_fix1234
```

If A0 found it above 0.95, run the same command **without** `--init-from $CK/step_050000`.

**Expected time:** about 12 minutes. This adds F4, using `λ ← max(0, λ + gap)` with η=1. An alarm saves a checkpoint and exits with code `3`.

## Compare A1–A3 logs

```bash
$PY experiments/summarize_qrl_log.py --log $D/v08/check/a2_fix123/train_metrics.jsonl \
  --compare $D/v08/check/a3_fix1234/train_metrics.jsonl $D/v08/check/a1_reproduce/train_metrics.jsonl
```

The command prints rows at 5k-step intervals and the alarm step(s) for each run. Send the stdout plus the three `train_metrics.jsonl` files and the `training.jsonl` held-out reports for interpretation.

### Gate for choosing A2 or A3

At the 20k end point, a setting passes only if all conditions hold:

- No F3 collapse alarm occurred.
- The random-pair mean `d(s,g)` is not trending downward in the 5k summary.
- Held-out one-step violation rate `d(s,s′) > 1.25` is at most **0.223**.
- Held-out `P(d≤5 | same-episode j=5)` is at least **0.623**.

If neither A2 nor A3 passes, do not start a 200k run. Return the logs for review. If one or both pass, compare one-step violation rates; the lower-rate passing setting is used for the later P13 and standard 4×4 runs. Those 200k commands will be added after reviewing these outputs, as specified in v0.8 §2.2.

**Expected time for this stage:** roughly 40 minutes total for tests, A0, and A1–A3, excluding setup and output review.
