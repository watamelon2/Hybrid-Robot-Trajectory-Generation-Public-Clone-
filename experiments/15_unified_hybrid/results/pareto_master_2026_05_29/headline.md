# Post-hoc consolidated Pareto / stability analysis

Generated 2026-05-29 by `aggregate_pareto_master.py` over 36 measured cells across 6 slate directories on `main` (post merge cascade).

## Cells on the (adapt_wall_s, succ_rate) Pareto front

4 cells survive the upper-left Pareto filter:

| method | tag | succ | Wilson 95 % CI | adapt_wall_s |
|---|---|---:|---:|---:|
| hybrid_canonical | T2_0_hybrid_canonical_bank_mass1p3_seed2 | 22/30 = 73.3% | [55.6, 85.8] | 0.0 |
| hybrid_5 | T2_1_hybrid_5demo_adapted_mass1p2 | 24/30 = 80.0% | [62.7, 90.5] | 1.0 |
| hybrid_other | T2_1_lambda_2p0 | 24/30 = 80.0% | [62.7, 90.5] | 1.0 |
| hybrid_other | T2_1_lambda_5p0 | 24/30 = 80.0% | [62.7, 90.5] | 1.0 |

## T2.1 stability across perturbation knobs

- **Anchor T2.1 (mass=1.3, 5 perturbed demos, lambda=1, seed=0):**  22/30 = 73.3%  [55.6, 85.8].
- **seed (multiseed):**  2 of 2 cells inside T2.1's Wilson CI.
- **mass (magnitude sweep):**  4 of 4 cells inside T2.1's Wilson CI.
- **n_perturbed (democount):**  4 of 4 cells inside T2.1's Wilson CI.
- **lambda (lambda sweep):**  2 of 3 cells inside T2.1's Wilson CI.  Outside-CI cells: T2_1_lambda_0p5 (30.0%)

*Interpretation.*  A knob "passes" the stability check when most of its cells land inside the anchor's Wilson 95 % CI; this means the T2.1 headline is not specific to a particular choice of that knob and would survive minor design changes.
