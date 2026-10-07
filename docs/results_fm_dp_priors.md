# Standalone performance — Flow-Matching (FM) and Diffusion-Policy (DP) priors

*Common-protocol re-measurement, 2026-06-04. Source slate:
`experiments/08_flow_matching_prior/slate_headline_priors_2026_06_04.sh`.
Raw JSONs:
`experiments/08_flow_matching_prior/results/headline_priors_2026_06_04/`.*

## 1. What this measures

Each prior is run **standalone** (no cost guidance, `λ_prior` off — the
model generates the action trajectory directly) on the `can` task, so the
number is the *raw generative quality* of the trained prior, not the
hybrid. All three priors were evaluated on **one identical protocol**:

- **50 fresh test seeds**, `test_start_seed = 100000` (disjoint from any
  training/selection seeds).
- `max_env_steps = 400`, `sigma ∈ [0.05, 0.25]`.
- **Per-prior native sampler config** (forcing a common K/chunk stalled
  the flow priors, so each runs at the operating point it was designed
  for — see col. notes).

## 2. Headline numbers

| Prior | Arch / backbone | Obs | Sampler | Chunk (n_action) | Test succ | n | Wall |
|---|---|---:|---:|---:|---:|---:|---:|
| **DP `ep0199`** | `dp_unet1d` (DDPM ε) | 23 (`can_dp`) | K=100 | 8 (closed-loop H=16) | **76.0 %** | 38/50 | 45 min |
| **v6 CFM** | `cfm_v6` / conv1d-UNet | 46 | K=18 (Heun) | 60 (open-loop) | **56.0 %** | 28/50 | 3.5 min |
| **v3 EDM** | `karras_edm_v3` / FiLM-MLP | 46 | K=18 | 60 (open-loop) | **0.0 %** ⚠ | 0/50 | 1.8 min |

## 3. Reading the results

- **DP is the strongest standalone prior — 76 %.** This is the canonical
  `v3b ep0199` checkpoint run in its native closed-loop config (T_obs=2,
  H=16, replan every 8, full K=100 DDPM). It is the *trained upper
  reference* in our story. (76 % here vs the 80 % quoted from the
  phase-B selection sweep is within seed noise across the two 50-seed
  buckets; both are below the published low-dim DP crosswalk of 94.7 %
  for the protocol/budget reasons in
  `docs/priors_design_choices.md` §F.)
- **v6 CFM = 56 %.** A real, usable flow prior, but ~20 pp below DP. It
  runs ~13× faster (3.5 min vs 45 min) because it samples in K=18 Heun
  steps over a long open-loop chunk (60) rather than K=100 closed-loop.
  The previously-quoted **83.3 % CFM was on the easier 30 train-init
  grid**, not these 50 fresh test seeds — so 56 % is the honest,
  apples-to-apples figure and supersedes 83.3 % for cross-prior
  comparison.
- **v3 EDM = 0 % — a persistent, reproduced result, not a one-off
  glitch.** The FiLM-MLP EDM checkpoint (`v_theta_v3.pt`) has scored
  **0 %** in *every* evaluation to date, not just this run:
  `v3_eval_first` (λ=0.0 and λ=1.0, n=60 each) and `v4_unet_standalone`
  (`karras_edm_v3`) are all 0/N. Root cause is **unresolved** — it is
  either a systematic fault in the EDM score/sampler integration that
  has never been fixed (e.g. wrong operating point / normaliser /
  Tweedie sign at runtime) **or** a genuinely non-functional FiLM-MLP
  EDM checkpoint. **No v3 number — standalone or hybrid — should be put
  in any headline figure until this is root-caused.**

## 4. Why DP > FM here

The gap is consistent with the design differences documented in
`docs/priors_design_choices.md`:

- **Capacity:** DP uses a ~64 M-param conv1d-UNet; v3 uses a small
  FiLM-MLP. v6 (conv1d-UNet) closes much of that gap (56 % vs DP's 76 %).
- **Closed- vs open-loop:** DP replans every 8 steps from a 2-step obs
  history (error-correcting); the FM priors commit to a 60-step open-loop
  chunk and cannot react to drift. This is the single biggest lever.
- **Sampler budget:** DP spends K=100 ancestral steps; the FM priors use
  K=18. Cheaper, but lower fidelity for an under-EMA'd model.

## 5. Hybrid results (prior + MBD score-bridge) for FM / DP

Standalone quality is *not* the deliverable — in the hybrid the prior is
run in **`pull` mode** (trained-prior-as-retrieval) and combined with the
MBD cost score at strength `λ_prior`. These numbers are from their own
slates (`can/ph`, K=10, closed-loop), **n=30 unless noted, different seed
bands — do not pool with the 50-seed standalone table in §2.**

### 5.1 DP prior + bridge (Phase-D grid)

Source: `experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.md`.
The lift is **operating-point dependent** — it appears only at the
moderately-trained checkpoint:

| DP ckpt | λ=0.1 | λ=0.5 | λ=2.0 | K=20, λ=0.5 |
|---|---:|---:|---:|---:|
| `ep0049` (under-trained) | 0.475 | 0.550 | 0.525 | 0.533 |
| **`ep0149` (Q1 peak)** | 0.600 | 0.733 | **0.867** (26/30) | **0.833** (25/30) |
| `ep0199` (strongly-trained) | 0.733 | 0.717 | 0.733 | 0.700 |

- **Headline DP hybrid = 86.7 %** [70.3, 94.7] at `ep0149`, K=10, λ=2.0
  (+13.4 pp over the λ=0.5 baseline); K=20, λ=0.5 corroborates at 83.3 %.
- The **`ep0149` row is uniquely λ-monotone** (0.60 → 0.73 → 0.87); the
  under-trained and strongly-trained checkpoints are flat under every
  (K, λ). Mechanism: MBD adds value only when the prior's score field is
  coherent *and* there is headroom above the prior's standalone rate.
- pure-MBD (no prior) on the same task = **0 %**.

### 5.1.1 Checkpoint protocol note — `ep0199` vs `ep0149`

These two checkpoints play different roles and should not be substituted
for one another:

- **Standalone DP reference:** `ep0199`. This is the canonical promoted
  DP checkpoint for the common-protocol standalone table (§2), where it
  scores 76 % on the June-04 bucket and 80 % in the Phase-B selection
  sweep. Use `ep0199` whenever the claim is "how strong is DP by itself?"
- **Hybrid-lift operating point:** `ep0149`. This is the checkpoint where
  the Phase-D bridge grid finds a positive MBD lift: 86.7 % at K=10,
  λ=2.0 and 83.3 % at K=20, λ=0.5. Use `ep0149` only for the claim
  "MBD helps a moderately-trained prior with headroom."

The correct interpretation is therefore **not** "replace `ep0199` with
`ep0149`." It is: `ep0199` is the stronger standalone DP reference, while
`ep0149` is the better hybrid-lift point. Any future T8/DP-prior
replication should state which claim it is testing before choosing the
checkpoint.

### 5.2 CFM (v6) prior + bridge (`pull` mode)

Source: `experiments/08_flow_matching_prior/results/eval_dp_ema_e300/`
(`v6pull`, K=10, R=1, n=30). The bridge is the dominant lift here too:

| λ_prior | Succ | n | Rate |
|---:|---:|---:|---:|
| 0.0 (warm-start only) | 5 | 30 | **16.7 %** |
| 0.25 | 15 | 30 | **50.0 %** |
| 2.0 | 21 | 30 | **70.0 %** |

CFM warm-start alone is weak (16.7 %); the score-bridge carries it to
70 % at λ=2.0 — the same "bridge does the work" pattern as DP.

### 5.3 v3 EDM prior + bridge

**No usable hybrid number.** The standalone v3 cell is currently 0 %
(§3), so the hybrid is not meaningful until that fault is fixed. (The
73.3 % Phase-F headline in earlier docs was the *DP* prior, not v3.)

### 5.4 Reference point

Zero-train `demo_1nn` + bridge = **73.8 %** [67.8, 78.9] at *0* trained
parameters (`slate_e_2026_05_19/AGGREGATE.json`).

**The story:** the cheap zero-train retrieval prior already reaches
73.8 %; the trained DP prior + bridge tops it at 86.7 % but *only* at the
right checkpoint and λ; CFM + bridge reaches 70 %. The **bridge is what
lets any prior add value** — every prior's λ=0 warm-start is far weaker
than its bridged number.

## 6. Caveats / open items

- v3 EDM 0 % is **reproduced across every eval** (this run +
  `v3_eval_first` + `v4_unet_standalone`), so it is a persistent
  EDM-integration/checkpoint problem, not a one-off — root-cause before
  any figure update.
- DP 76 % vs 80 % is cross-bucket seed noise (n=50 each, ±~6 pp at 1σ);
  do not over-interpret.
- v6 56 % supersedes the old 83.3 % only because the protocol changed;
  re-running the old hybrid cells on this 50-seed protocol is the
  follow-up before the slide is finalised.

## 7. Cross-references

- Prior design choices & reproduction-gap analysis:
  `docs/priors_design_choices.md`.
- Published SOTA reference numbers: `docs/references/sota_reference_numbers.md`.
- Figure generator (pending update with these numbers):
  `presentation/scripts/make_headline_figures.py`.
