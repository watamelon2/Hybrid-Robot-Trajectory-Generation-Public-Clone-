# Phase 3 — Deep-dive notes (2026-06-11 refresh)

Notes focus on methodology, experiments, and the explicit connection/threat to
HRTG's planned next experiment. Read via abstracts + ar5iv section structure;
re-verify exact numbers from PDFs before camera-ready.

## 1. DynaGuide — Steering Diffusion Policies with Active Dynamic Guidance [@bagaria2025dynaguide]

**Problem.** Deployed policies must be *steered* to situation-specific objectives
without retraining the base policy with those objectives in mind.

**Method.** Keep a frozen, pretrained action-diffusion policy. Train a *separate*
dynamics model that predicts future outcomes from candidate action chunks. During
the denoising of each action chunk, add a guidance gradient from this dynamics
model toward the desired outcome — i.e., classifier-style guidance from an external
model, decoupled from the base policy. The decoupling lets one base policy be
steered to multiple/changing objectives and stay robust when the objective signal
is low quality.

**Experiments.** Complex 3D simulated tasks; a robustness study under low-quality
guidance; multi-objective steering; enhancing underrepresented base behaviors; and
real-robot steering of an off-the-shelf policy. Headline: ~70% steering success on
articulated CALVIN tasks and **5.4× over goal-conditioning when steered with
low-quality objectives**.

**Limitation / open question.** Requires training a dynamics/outcome model; guidance
strength is a manual knob; no predictor for *when* guidance helps vs hurts on a
given task/prior.

**Connection / threat to HRTG.** This is the **closest method**: HRTG's
`lambda_prior` cost-guided denoising of a frozen prior (retrieval or DP) is the same
operation with the MBD cost playing DynaGuide's dynamics-model role. DynaGuide
optimizes *steering success*; HRTG's contribution must be the **calibration boundary**
— predicting *when/how-strongly* to guide and explaining *why product/gate rules
over-optimize* — which DynaGuide leaves as a manual knob. HRTG also covers the
zero-training retrieval prior, which DynaGuide does not.

## 2. ADPro — Test-time Adaptive Diffusion Policy via Manifold + Initial-Noise Constraints [@adpro2025]

**Problem.** Vanilla DP treats action generation as unconstrained denoising,
ignoring geometry/control structure available a priori.

**Method.** Two test-time inductive biases, no retraining: (1) a **geometric manifold
constraint** that aligns denoising updates with task-relevant subspaces (using
end-effector↔target relative pose as a natural gradient direction along a geodesic),
and (2) an **analytic initialization** that proposes a structured initial noisy
action from a rough gripper↔target registration instead of an uninformative prior.
Compatible with pretrained DPs.

**Experiments.** RLBench, CALVIN, and real-world data; ablations isolate the manifold
constraint and the init. Reports generalization gains over DP baselines.

**Limitation / open question.** Constraints are task-geometry specific; like
DynaGuide, no principled criterion for *when* the adaptation helps a given prior.

**Connection / threat to HRTG.** ADPro is an **alternative combination rule** to
HRTG's additive score-bridge (manifold projection + smart init vs additive cost
gradient). It strengthens the case that "test-time correction of a pretrained DP"
is a real, active design space — and that the open question is *which rule, and when*.
HRTG's matched-seed comparison of additive vs product/gate/temper is precisely a
study in that space. Cite ADPro as a contemporary alternative, and note HRTG's
init-anchor (`pull`) and additive bridge are simpler and training-free.

## 3. Calibrated Test-Time Guidance for Bayesian Inference [@calibrated2026]

**Problem.** Test-time guidance steers pretrained diffusion toward a reward, but
does it sample the correct posterior?

**Method.** Shows common guidance methods **do not** recover the true Bayesian
posterior; identifies the structural approximations (e.g., the Gaussian/score
approximation of the guidance term) that cause miscalibration; proposes consistent
estimators that restore calibrated sampling.

**Experiments.** Bayesian inference tasks + black-hole image reconstruction; beats
prior guidance methods on calibration metrics.

**Limitation / open question.** Not robotics; calibration measured on inference
tasks, not control success.

**Connection to HRTG.** This is the **formal theory hook**. HRTG's empirical finding
— additive score-bridging helps while product/gate/temper over-optimize and fail — is
a control-domain instance of "guidance is miscalibrated unless the combination rule
matches the prior geometry." HRTG can cite this to argue its calibration/diagnostic is
the applied robot-manipulation counterpart of a known theoretical problem.

## 4. Test-time Alignment of Diffusion Models without Reward Over-optimization (ICLR 2025) [@das2025]

**Problem.** Aligning a diffusion model to an objective at test time without either
over-optimizing (fine-tuning) or under-optimizing (naive guidance).

**Method.** Training-free Sequential Monte Carlo sampling from the reward-aligned
target with tempering; samples the tempered posterior instead of greedily maximizing
reward. Code: `github.com/krafton-ai/DAS`.

**Experiments.** Single-reward, multi-objective, and online black-box optimization;
matches/beats fine-tuning on target reward while preserving diversity.

**Connection to HRTG.** Names HRTG's exact failure mode: greedy reward-maximizing
combination (product/gate) **over-optimizes** and collapses, while a tempered/additive
mix preserves the prior's diversity/coherence. Supports the recommendation that the
*combination rule*, not the λ value, is the primary knob — and motivates reporting
HRTG's additive bridge as the well-behaved operating point.

## 5. DARP — Difference-Aware Retrieval Policies for Imitation Learning [@darp2026]

**Problem.** BC compounds errors out-of-distribution; reusing training data at
inference can mitigate this.

**Method.** Semi-parametric: instead of a global state→action map, **train** a model
to predict actions from the k nearest expert neighbors, their actions, and the
relative distance vectors between neighbor states and the query state. No extra data,
no online expert, no task-specific knowledge beyond standard BC assumptions.

**Experiments.** Continuous control + robotic manipulation, multiple representations
(incl. high-dim visual); 15-46% over BC. Project: `weirdlabuw.github.io/darp-site`.

**Limitation / open question.** Still trains a (difference-aware) model; retrieval is
used to condition a learned policy, not as a zero-training controller.

**Connection / threat to HRTG.** Closest neighbor to `demo_1nn`. **Differentiation:**
HRTG uses retrieval with **zero training** as a *score/pull target inside a test-time
optimizer (MBD)*, not as conditioning for a trained policy. HRTG's distinct claims are
(i) retrieval-as-test-time-score-prior and (ii) the bank-size finite-coverage curve.
DARP must be the headline comparison in HRTG's related work for the retrieval claim.

## 6. Hierarchical Diffusion Policy via Contact Guidance (T-RO) [@hdp2025]

**Problem.** End-to-end DP underperforms on contact-rich tasks and lacks
interactivity/interpretability.

**Method.** Two-level policy: high-level predicts the next **contact point** from 3D
info; low-level predicts an action sequence toward that contact, conditioned on
observation + contact latent. Both are conditional denoising diffusion; low level
combines BC + Q-learning to drive actions toward contact. Adds a one-shot gradient
guidance scheme.

**Experiments.** Six tasks; **+20.8% average over Diffusion Policy**, largest gains on
contact-rich tasks; reports improved interpretability/interactivity.

**Connection to HRTG.** Direct template for the **phase/contact** direction:
`tool_hang`/Square residuals are contact-rich phase failures. Suggests adding a
high-level contact/subgoal signal and running short low-level guided segments rather
than one long sparse objective. Higher impact but higher implementation/compute risk.

## 7. Reactive Diffusion Policy [@xue2025reactive]

**Problem.** Contact-rich tasks need fast reaction to tactile/force feedback;
action-chunking DP cannot react within a chunk.

**Method.** Slow-fast hierarchy: a slow latent diffusion policy for high-level chunks
+ a fast asymmetric tokenizer for closed-loop tactile/force control; AR teleop system
for force feedback data.

**Experiments.** Three contact-rich tasks; significant gains over visual IL.

**Connection to HRTG.** Evidence that Square/`tool_hang` residual failures are
contact-observability failures. In HRTG's low-dim Robomimic setting, the analogue is
adding peg/nut-relative pose and contact proxies to the trace before more λ sweeps.

## 8. Beyond Binary Success — SAVI Robot Policy Comparison [@savi2026]

**Problem.** Robot evaluation is rollout-limited and usually reduced to binary
success, which is statistically weak.

**Method.** Safe, anytime-valid inference (SAVI): a sequential test that can stop early
once evidence reaches a confidence level; handles partial-credit progress, continuous
reward, trajectory smoothness; parametric and nonparametric.

**Experiments.** Sim + real eval data; up to **70% fewer rollouts** than batch methods,
50% fewer than prior sequential binary methods, no loss of rigor.

**Connection to HRTG.** A cheap rigor upgrade: HRTG cells are n=30 with Wilson CIs.
Guided-vs-unguided lift can be tested with anytime-valid sequential tests and
partial-credit metrics (e.g., Square task-progress), reducing required rollouts —
useful given finite remote compute.

## 9. Diffusion Policy [@chi2023diffusion]

**Problem.** BC struggles with multimodal, high-dim continuous actions.

**Method.** Conditional denoising diffusion over action sequences; iterative
refinement at inference.

**Experiments.** 12 tasks / 4 benchmarks; large gains over prior visuomotor baselines.

**Connection to HRTG.** The canonical prior HRTG guides. HRTG's Can-DP and Square-DP
results are test-time hybridizations of this prior; the checkpoint-regime story
(ep0149 hybrid-lift vs ep0199 standalone) is about *how trained* this prior is.

## 10. Neural-Dynamics-Augmented Diffusion Policy [@ndadp2025]

**Problem.** Robust DP needs many demos; model-based planning covers configurations
but struggles without demo guidance.

**Method.** Train a robust DP in a **local support region** from 1-30 demos, and use an
offline-trained neural dynamics model to *rearrange* out-of-region configurations back
into the supported region.

**Experiments.** Granular, contact-rich, multi-object tasks in sim + real; with 1-30
demos, covers a much larger region than demo-only policies.

**Connection to HRTG.** Supports the **coverage/adaptation** framing: HRTG's bank-size
curve and demo-bank augmentation are the retrieval analogue of "expand the supported
region cheaply." kNN distance to the bank is HRTG's "in-region" signal.

## 11. Robot Data Curation with Mutual-Information Estimators [@curation2025]

**Problem.** Demonstration *quality* is rarely measured despite its importance.

**Method.** Estimate each trajectory's contribution to the state-action mutual
information using a **kNN MI estimator** over VAE embeddings; partitions datasets by
quality matching human expert scores.

**Connection to HRTG.** Provides a principled, kNN-based **bank-coverage/quality
diagnostic** — exactly what HRTG's retrieval-coverage direction needs to predict when a
few demos will help (locality/disagreement metrics).

## 12. Policy Gradient Guidance [@pgg2025]

**Problem.** Can classifier-free guidance be ported beyond diffusion?

**Method.** Add an unconditional branch to policy gradients and interpolate
conditional/unconditional (guidance scale γ). Theory shows the extra normalization
vanishes under advantage estimation.

**Experiments.** Discrete + continuous control. Finds conditioning **dropout
destabilizes continuous control**, while modest γ>1 improves stability/sample
efficiency.

**Connection to HRTG.** Independent evidence that guidance is regime-dependent and can
*destabilize* continuous control — consistent with HRTG's negative results and the
calibration thesis: more guidance is not monotonically better.
