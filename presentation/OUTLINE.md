# Week-8 Presentation — Slide Outline & Speaking Script

*6 min talk + 3 min Q&A, hard stop at 7 min.  9 slides → ~40 s
each; the two results slides get the most airtime.*

Speakers: Zihao Sun (slides 1–5), Charles Beck (slides 6–9).
Adjust the split however suits us.

Each slide block below has four parts:
1. **Sketch** — visual layout, what is on screen.
2. **Materials** — which figures / videos / code to render, where
   they live in the repo.
3. **Script** — verbatim words, ~150 wpm = ~150 words per minute.
4. **Timing** — target seconds.

---

## Slide 1 — Title (10 s)

**Sketch.**  Single centered title card.  Bottom-right corner:
ECE M237 logo / UCLA wordmark optional.

```
Hybrid Robot Trajectory Generation
Learned Demonstration Priors meet Model-Based Diffusion

Zihao Sun · Charles Beck
ECE M237 · Spring 2026
```

**Materials.**  None — pure title text.  Background: one still
frame from a successful PickPlaceCan rollout
(`presentation/figures/hero_can_grasp.png`, render with
`scripts/render_rollouts.py --prior demo_1nn --task can --seed 0
--save-still`).

**Script.**
> "Hi, we're Zihao and Charles.  Our project is *Hybrid Robot
> Trajectory Generation*: we combine a learned demonstration
> prior with the analytical score of model-based diffusion in a
> single denoising loop, and we ask whether the combination
> crosses the binary success threshold on robomimic manipulation
> where neither component succeeds alone."

**Timing.** 10 s.

---

## Slide 2 — Motivation (45 s)

**Sketch.**  Three-panel comparison on top half:

| Behaviour cloning | Model-based diffusion (MBD) | Our hybrid |
|---|---|---|
| Demo video, BC drifts | Demo video, MBD jitters | Demo video, hybrid grasps |
| `bc_failure.mp4` | `mbd_failure.mp4` | `hybrid_success.mp4` |

Bottom half:  the two failure modes in one sentence each, then a
question mark icon and the headline:

> "Can a *learned* prior **rescue** a model-based planner — and
> does the cost function still get to add anything?"

**Materials.**
- `videos/bc_failure.mp4`     — BC rollout on `can`, no-reach
- `videos/mbd_failure.mp4`    — pure MBD rollout, jittery, 0/60
- `videos/hybrid_success.mp4` — demo-1NN+MBD rollout, clean grasp

  All three rendered by
  `scripts/render_rollouts.py --comparison-trio`.

**Script.**
> "Two classical failure modes motivate this work.  Behaviour
> cloning gives you a smooth policy but it doesn't recover from
> distribution drift — on PickPlaceCan it succeeds 7 % of the
> time, mostly because it never actually reaches the can.
> Model-based diffusion does the opposite — it has a cost
> function it can re-evaluate, but with a random initialisation
> on a 120-step, 7-DoF problem, the search just doesn't find
> the basin: 0 % on Can.  The interesting intellectual question
> is whether you can compose them — give the planner a *learned*
> prior over what good trajectories look like, and let the
> physics-based cost prune the rest.  That's what this talk is
> about."

**Timing.** 45 s.

---

## Slide 3 — Problem Formulation (50 s)

**Sketch.**  Top: one equation, displayed.  Bottom: a 3-row
table of "what each symbol is".

Displayed equation:

```
∇ log p_t(τ)  ≈  ∇ log p_data,t(τ)  +  λ(σ_t) · ∇ log p_model,t(τ)
                ───────────────                ───────────────
                  learned prior                   MBD cost
```

with `p_model ∝ exp(−J(τ)/β)` next to it as a footnote.

Below the equation, three labelled rows:

- $p_\text{data}$: learned from demos (BC / flow / retrieval / DP).
- $p_\text{model}$: hand-designed cost $J(\tau)$ (goal-distance,
  contact, grip).
- $\lambda(\sigma_t)$: noise-schedule-dependent weight that
  blends them — $\lambda=0$ recovers MBD, $\lambda\to\infty$
  recovers pure prior denoising.

**Materials.**  Render the equation in TikZ / LaTeX directly in
Keynote/Beamer.  No external file needed.

**Script.**
> "Concretely we replace the score in the denoising chain with a
> composite of two scores: one from a *learned* prior over demo
> trajectories, one from the analytical gradient of a
> hand-designed cost.  Lambda controls the mix — lambda equals
> zero recovers the original model-based diffusion; lambda goes
> to infinity recovers pure prior denoising.  The interesting
> region is between them.  The Boltzmann temperature beta picks
> the cost's softness."

**Timing.** 50 s.

---

## Slide 4 — Method & Comparisons (60 s)

**Sketch.**  Left column: a flow-diagram of one MBD iteration

```
  τ_t ─┐
       ├─▶ M perturbations  ─▶ J(τ̃_t^m) ─▶ SNIS weights
       │                                       │
  prior τ^prior ──▶ pull (−λ·(τ_t−τ^prior)/σ²)─┘
                                               │
                                               ▼
                                         τ_{t−1}
```

Right column: the comparison table — which baseline tests what.

| Prior family | What it tests | Best @ Can/ph |
|---|---|---|
| BC (MLP) | smooth policy alone | 0.067 |
| Warm-start ablation | structured init only | 0.017 |
| MBD alone | cost-only search | 0.000 |
| Flow-matching v1 / v2 | offline generative prior | 0.000 / 0.167 |
| **Retrieval (1-NN, zero-train)** | nearest-demo anchor | **0.738** |
| **Trained DP (CFM action chunks)** | full diffusion-policy prior | **0.733** standalone |

Bold rows are the surviving baselines; everything else falls
out of the comparison after the first results slide.

**Materials.**  Flow diagram: hand-drawn in slide tool; table
data lives in `docs/results.md` and `experiments/17/results/`.

**Script.**
> "The method has one update rule.  Every MBD iteration draws
> M perturbations of the current trajectory; we evaluate the
> cost on each, weight them by SNIS, *and* we add a Gaussian
> pull toward the prior with strength lambda over sigma
> squared.  When sigma is large the pull dominates; when it
> shrinks the cost takes over.  We swap the prior across four
> families: behaviour cloning, an offline flow-matching
> generative model, a zero-training 1-NN retrieval against the
> demo bank, and a trained diffusion policy.  These are the
> baselines we'll compare against on the next slide."

**Timing.** 60 s.

---

## Slide 5 — Result 1: Cross-task Headline (70 s)

**Sketch.**  Large bar chart in the centre:

x-axis: `lift`, `can`, `square`
y-axis: success rate
4 grouped bars per task: BC / MBD / Retrieval / Trained-DP-Hybrid.
Keep SOTA references off the chart for readability.

Wilson 95 % CI error bars on every bar.  Annotate retrieval-on-
lift = 1.000 with a star.

Right margin: callout box

> Retrieval + MBD reaches **73.8 %** on Can
> (4-seed pool, n = 240, CI [68, 79]) and solves Lift.
> The green DP bars are reference cells: Can rises to 86.7 %,
> but the zero-training retrieval headline remains the main claim.
> Square is partially unlocked at 50 % with `H=300, λ=2`.

**Materials.**
- `figures/headline_bar_crosstask.pdf` — bar chart.  Generate
  with `scripts/make_headline_figures.py --plot crosstask`.
- Numbers come from
  `experiments/15_unified_hybrid/results/slate_*/AGGREGATE.json`
  and `experiments/17_dp_hybrid/results/phase_d_2026_05_30/q1_AGGREGATE.json`.

**Script.**
> "This is the cross-task headline.  On Lift and Can, zero-training
> retrieval plus the MBD bridge already gets the main result: Lift
> solves entirely, and Can reaches seventy-four percent over a
> four-seed pool.  The green DP bars are reference cells — they show
> the costlier trained-prior path can push Can to eighty-seven
> percent.  Square is no longer a pure floor: at H equals 300 and
> lambda equals 2, retrieval reaches fifty percent."

**Timing.** 70 s.

---

## Slide 6 — Result 2: Two GREEN Cells at the Prior-Quality Peak (80 s)

**Sketch.**  Two stacked plots side by side.

Left: **λ-slice at ep0149** (cheaper-path GREEN).
- x-axis: λ ∈ {0.1, 0.5, 2.0} (log scale)
- y-axis: success rate, with 95 % CIs
- Three points, monotone climb 60 % → 73.3 % → **86.7 %**
- Star on (λ=2.0, 86.7 %); annotate "+13.4 pp vs canonical, *same compute*"

Right: **K-slice at ep0149** (corroborating GREEN).
- x-axis: K ∈ {10, 20}
- y-axis: success rate, 95 % CIs
- Lift from 73.3 % → **83.3 %**; annotate "+10.0 pp"

Below both plots, one sentence:

> MBD lifts iff the prior score is informative and the prior
> still has headroom.  The full heatmap is backup/Q&A; the main
> slide should show the two clean GREEN cells.

**Materials.**
- `figures/green_slices.pdf` — generated by
  `scripts/make_headline_figures.py --plot green_slices`.
- Source data: `experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.json` (Phase E + F numbers).
- Keep `figures/heatmap_ckpt_K.pdf` as backup/Q&A only.

**Script.**
> "We then asked: where does the lift live?  Two knobs.  Left
> panel — at ep one-forty-nine, the moderate-prior peak,
> sweeping the bridge weight λ gives a *clean monotone climb*
> from sixty to eighty-seven percent.  This is the **strongest
> cell in the entire grid** — a thirteen-point lift over the
> baseline at the *same* K-equals-ten compute.  Right panel,
> the same prior under a larger K-equals-twenty refinement
> budget also clears the bar at eighty-three percent, ten
> points up.  Two independent knobs both move ep0149 into
> GREEN.  The lift requires a coherent score field *and*
> headroom — both conditions, neither alone is enough.  The
> cheaper bridge-weight knob is also the bigger lift, which is
> the surprise."

**Timing.** 80 s.

---

## Slide 7 — Result 3: Adaptation — Bank-Size Scaling Law (45 s)

**Sketch.**  Single curve plot:

- x-axis: bank size N_b ∈ {5, 10, 20, 50, 100, 200} (log scale)
- y-axis: success rate, 95 % CIs
- Three-phase shape annotated on the curve:
  - "under-coverage" (5 → 35 %)
  - "knee" (20 demos → 65 %)
  - "plateau" (50 → 200 flat at 68-70 %)

Below, callout:

> The prior is **coverage-limited, not size-limited**.  Tripling
> the bank past N_b = 20-50 buys nothing on Can.  This supports
> the zero-training claim: the full 200-demo bank is not a hidden
> memorization requirement; most of the gain is finite mode coverage.

**Materials.**
- `figures/scaling_law.pdf` — regenerated from the June-7
  bank-size slate by `presentation/scripts/make_headline_figures.py`.

**Script.**
> "One adaptation experiment.  We swept the retrieval bank from
> five demos to two hundred.  The curve has three phases:
> twenty-three, thirty-five percent under coverage; then the
> knee at twenty demos, sixty-five percent; then a *dead-flat
> plateau* from fifty to two hundred demos.  The prior is
> coverage-limited, *not* size-limited — the zero-training
> headline is not a full-bank memorization artifact."

**Timing.** 45 s.

---

## Slide 8 — Lessons Learned & What Didn't Work (50 s)

**Sketch.**  Two-column "wins / lessons" layout.

Wins
- Composite score works *only* at moderate-prior + coherent
  score + headroom.
- Zero-training retrieval is a remarkably strong baseline.
- Bank-size scaling law is clean and theory-matching.

What didn't work / lessons
- Flow-matching EDM failed across every honest eval; CFM is usable
  but still below retrieval + bridge.
- Tempered / gated product-vs-max variants did not survive
  replication; additive score-bridge is the robust operating mode.
- Square is **partially unlocked** by `H=300, λ=2` (50 %), but
  needs replication and failure-phase localization before it becomes
  a headline solved task.
- Phase-C wrapping bug: chunk-boundary `env.reset()` wiped OSC
  state and silently zeroed the MBD-only sanity gate.  Two
  weeks of "MBD doesn't help" was infrastructure, not algorithm.

**Materials.**  Text only.  Optional small icon callouts.

**Script.**
> "Lessons.  First, the composite-score story is real but
> *narrow* — it only lifts when both conditions hold.  Second,
> zero-training retrieval is a surprisingly strong baseline; we
> spent a lot of compute training priors that
> rarely beat retrieval once measured honestly.  Third — and this one stung — a silent
> infrastructure bug at the MBD chunk boundary cost us two
> weeks before we noticed.  The pure-MBD sanity gate was
> dropping to zero out of fifty because robosuite's
> operational-space controller state was being wiped on
> every reset; gating the reset on K > 0 restores it to
> eight out of ten.  The takeaway: instrument every layer of
> the wrapper stack from day one."

**Timing.** 50 s.

---

## Slide 9 — Future Work & Thanks (30 s)

**Sketch.**  Bullet list of in-flight work, then a thank-you
line.

Next research directions
- Replicate Square `H=300, λ=2`, sweep nearby λ, and log failure phase.
- Move tool_hang / transport to learned value or subgoal signals;
  more product/gate tuning is deprioritized.
- Repeat the adaptation story under a trained-policy prior.
- Turn bank-size saturation into a finite-coverage theory figure.
- Keep MJX as an engineering appendix: Gate-4 works, but 13.8×
  speedup is below the promotion threshold.

Bottom line:

> Thanks!  Questions?

**Materials.**  Pure text.

**Script.**
> "Six pieces of in-flight work are listed here, all on named
> branches.  The immediate next step is to replicate the new
> Square operating point and localize the failures; after that,
> tool-hang needs subgoals or value signals rather than more
> gate tuning.  Happy to take questions."

**Timing.** 30 s.

---

# Total budget

| Slide | s |
|---|---:|
| 1 Title | 10 |
| 2 Motivation | 45 |
| 3 Problem | 50 |
| 4 Method + comparisons | 60 |
| 5 Result 1 (cross-task) | 70 |
| 6 Result 2 (GREEN cells) | 80 |
| 7 Result 3 (adaptation) | 45 |
| 8 Lessons | 50 |
| 9 Future + thanks | 30 |
| **Total** | **440 s ≈ 7:20** |

That's slightly over budget.  Trim by ~30 s by tightening
slides 4 and 6 in rehearsal.

# Q&A prep (3 min)

Anticipated questions and one-line answers:

1. **"How is this different from Diffusion Policy?"** — DP is a
   pure prior; our composite score keeps DP's prior but adds an
   analytical cost gradient.  Phase F's λ=2.0 GREEN at K=10 with
   DP-as-prior is the direct demonstration.
2. **"Does Square work now?"** — partially.  At the old H=120
   operating point it was near zero, but H=300 λ=2 reaches 50 %.
   Horizon alone is not enough; next step is replication and
   failure-phase logging.
3. **"Where did the safety / CBF angle go?"** — scoped out per
   the Week-5 reviewer; the binary-success comparison is the
   single load-bearing claim now.
4. **"How sensitive to λ?"** — at ep0149, monotone in λ over
   {0.1, 0.5, 2.0}.  At other ckpts, flat.  Sensitivity is
   prior-quality-dependent.
5. **"Compute?"** — full Phase D grid (Q1 + Q2 + Q3 + E + F) ran
   on a single Mac via four chained slates and disjoint seed
   bands; GCP arm was killed mid-run by an orphan-worker bug.
