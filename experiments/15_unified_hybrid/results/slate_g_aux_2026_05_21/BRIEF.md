# Slate G-aux -- "crack Square" parallel companion (May 21, 03:40 PDT)

Launched alongside Slate G when the load-average snapshot showed
~25 % idle CPU (4 of 16 cores free).  Uses n-workers=5 so the
combined footprint with Slate G's n-workers=10 stays at 15/16
cores.  Independent of Slate G's screen state; runs in its own
detached screen `slate_g_aux`.

## Why these eight cells

Slate F (May 20) verdict: **Square is mechanism-limited with hard
1-NN.**  Two follow-up questions Slate F did not address:

1. **Is sigma=0.25 actually the F1 peak, or is the F2 4-point
   sigma-tail too coarse to see a sharper optimum?**  F2 jumped
   from sigma=0.05 (0/60) to sigma=0.25 (2/60) to sigma=0.40
   (1/60); a peak at sigma=0.10 or sigma=0.20 would be invisible
   at that resolution.
2. **Does a smoothed prior unlock Square where hard 1-NN didn't?**
   Slate G G2 sweeps soft-NN beta on Can.  Square's contact
   geometry is *more* multi-modal than Can's (square nut + peg
   has more discrete grasp configurations); the bias-variance
   optimum could sit at a different beta.  We've never tested
   non-1NN priors on Square at lambda=1.

AUX1 answers (1) at high resolution, AUX2 answers (2) at the F1
sweet spot.

## Tracks

All cells: K=10 R=1 M=16 H=120, seed=0, 20 demos x 3 rollouts =
60 rollouts/cell.

### AUX1 -- sigma-tail high-resolution (4 cells, ~2.5 h)

| sigma | rationale                          |
|------:|------------------------------------|
| 0.10  | between F2's 0.05 (0 %) and 0.25 (3.3 %) |
| 0.15  | midpoint                                   |
| 0.20  | left shoulder of the F2 peak               |
| 0.30  | right shoulder of the F2 peak              |

Acceptance:
- **Mechanism-limited harden**: all four cells <= 5 % succ
  (within F1 noise band).  Adds 240 rollouts to the "Square needs
  a structural change" argument.
- **F2 peak shifts**: some sigma in [0.10, 0.20] hits > 5 %.
  Updates the F1 sweet-spot to that sigma; mechanism is OK, F1
  was on the wrong sigma.

### AUX2 -- soft-NN beta-sweep on Square (4 cells, ~2.5 h)

| beta | what it tests                              |
|-----:|--------------------------------------------|
| 0.5  | strongly smoothed (close to uniform k-NN)  |
| 1    | mild smoothing                             |
| 2    | mode-leaning but still mixes 2-3 neighbours|
| 5    | near-hard 1-NN (high concentration)        |

We skip beta=0 (uniform k-NN: known to be variance-dominated and
generically weak per Slate D) and beta>>1 (recovers hard 1-NN
already measured in F1).

Acceptance:
- **Strong (mode-mixing unlocks Square)**: some beta in
  [0.5, 5] hits > 10 % succ -- a 3x improvement over hard 1-NN's
  3.3 %, which would *falsify* the F1 verdict and reopen Square.
- **Weak (intermediate band)**: peak in [5 %, 10 %].
  Mode-mixing helps modestly; not enough to declare Square
  unlocked but enough to motivate retrieval-side improvements
  (C1 contrastive embedding).
- **Falsification (locks mechanism limit)**: all cells <= 5 %.
  Soft-NN doesn't change the verdict.  Square's failure is in
  the bridge / planning model, not the prior identity.

## Engineering posture

- 8 cells, ~38 min/cell at n-workers=5 (vs ~22 min at n-workers=10
  in F2; the parallelism penalty is sub-linear because env
  construct + cache-warm amortise).  Total wall ~5 h.
- Bank cache for Square at action_history_len=0 was warmed by F1
  + F3; AUX1/AUX2 hit the cached path (no rebuild).
- No precondition on Slate G state.  If Slate G crashes mid-run,
  AUX continues independently and the results stand on their own.
- If load average climbs above 16 (would indicate oversubscription
  hurting both slates), kill the aux:
  `screen -X -S slate_g_aux kill`

## Source

- Launcher: `experiments/15_unified_hybrid/slate_g_aux_2026_05_21.sh`
- Output dir: `experiments/15_unified_hybrid/results/slate_g_aux_2026_05_21/`
- Branch: `feat/zero-train-priors`.
