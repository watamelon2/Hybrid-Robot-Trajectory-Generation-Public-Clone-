# Experiment 06 — Failure taxonomy (post-hoc analysis)

**Lifecycle:** **done**. A one-screen analysis script that consumes
any `run_can`-format JSON. No env, no MuJoCo; runs in <1 s.

**In-tree deep doc:**
[`experiments/06_failure_taxonomy/README.md`](../../experiments/06_failure_taxonomy/README.md).

## Purpose

Promote the informal "no-reach / partial-reach / strong-reach /
success" buckets used in exp 04/05 narration into a formal column
that every later experiment can fill in. Provides:

1. The per-method behaviour distribution.
2. An apples-to-apples **strong-reach → success conversion rate**
   (the most useful single number for the MPC ablation in exp 05b).
3. A markdown table ready to paste into a report.

## Buckets

| bucket          | criterion                                          |
|-----------------|----------------------------------------------------|
| `success`       | `success == True`                                  |
| `strong_reach`  | `final_exec_reward > +3.0`, not success            |
| `partial_reach` | `+0.5 < final_exec_reward ≤ +3.0`, not success     |
| `no_reach`      | `final_exec_reward ≤ +0.5`                         |

Thresholds match the published Exp 04 / 05 README tables
exactly.

## Reproduce

```bash
python experiments/06_failure_taxonomy/run.py
```

Defaults consume:
- `experiments/04_mbd_baseline/results/can_ph.json`
- `experiments/05_hybrid_method/results/can_ph.json`
- `experiments/05_hybrid_method/results/can_ph_mpc.json`

And writes:

| Path | Format |
|------|--------|
| `experiments/06_failure_taxonomy/results/taxonomy.json` | Machine-readable per-input rows. |
| `experiments/06_failure_taxonomy/results/taxonomy.md` | Markdown table for paste-into-report. |

Verified to reproduce exp 05b's 14.8 % strong-reach → success rate
and the 53 % / 12 % / 3 % no-reach collapse from exp 04 → 05 → 05b.

## Schema requirements

To extend the script to a new experiment's JSONs, the schema must
contain:

- `per_rollout[*].final_exec_reward` (float)
- `per_rollout[*].success` (bool)
- optional: `per_rollout[*].bc_exec_reward`, `bc_success`

Every `run_can`-format experiment (04, 05, 07, 15) emits this
schema, so post-hoc taxonomy is free.

## Files

| Path | Role |
|------|------|
| [`experiments/06_failure_taxonomy/run.py`](../../experiments/06_failure_taxonomy/run.py) | The whole experiment. Single-file ~60 LoC. |
| `experiments/06_failure_taxonomy/results/taxonomy.json` | Output rows. |
| `experiments/06_failure_taxonomy/results/taxonomy.md` | Output markdown. |

## Cross-references

- **Consumers**: any experiment that wants to report behaviour
  buckets. The exp 04 / 05 / 09 / 10 / 14 / 15 READMEs all quote
  numbers that originate here.
- **Pairs with**: [`14_seed_sweep.md`](14_seed_sweep.md) — its
  multi-seed pool feeds 9 JSONs into the taxonomy script.

## When to extend

The script accepts arbitrary `--inputs` paths. New rows for
later experiments are free:

```bash
python experiments/06_failure_taxonomy/run.py \
    --inputs \
      experiments/15_unified_hybrid/results/slate_e_2026_05_19/AGGREGATE.json \
      experiments/15_unified_hybrid/results/slate_h_2026_05_21/...H4_lift_1nn_hist1.json
```
