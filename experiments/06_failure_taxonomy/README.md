# Experiment 06 — Failure taxonomy (post-hoc)

**Status:** _done._ One-screen analysis script over the existing
04 / 05 / 05b JSONs. No env, no MuJoCo; runs in <1 s.

## Purpose

Promote the informal "no-reach / partial-reach / strong-reach /
success" buckets used in Exp 04/05 narration into a column that
every later experiment will fill in. Provides the per-method
behaviour distribution **and** an apples-to-apples
"strong-reach -> success" conversion rate, which was the most
useful single number to come out of Exp 05b.

## Buckets

| bucket           | criterion                                        |
| ---------------- | ------------------------------------------------ |
| `success`        | `success == True`                                |
| `no_reach`       | `final_exec_reward <= +0.5`                      |
| `partial_reach`  | `+0.5 < final_exec_reward <= +3.0`, not success  |
| `strong_reach`   | `+3.0 < final_exec_reward`, not success          |

Thresholds match the published Exp 04/05 README tables exactly.

## Reproducing the headline row

```bash
python experiments/06_failure_taxonomy/run.py
```

This consumes the three default inputs and writes:

- `results/taxonomy.json` — machine-readable rows (one per input).
- `results/taxonomy.md` — Markdown table for paste into a report.

Verified on this branch: the script reproduces the Exp 05b README's
"14.8 % strong-reach -> success" and the 53 %/12 %/3 % no-reach
collapse from Exp 04 -> 05 -> 05b.

## Extending

Pass any number of `--inputs` paths. The schema requires only:

- `per_rollout[*].final_exec_reward` (float)
- `per_rollout[*].success` (bool)
- optional: `per_rollout[*].bc_exec_reward`, `bc_success`

So every future experiment that follows the run_can result schema
gets a free taxonomy row.
