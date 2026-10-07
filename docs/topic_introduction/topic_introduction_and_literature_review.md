# Topic Introduction & Literature Review: Hybrid Model-Based and Data-Driven Score Fusion for Robot Trajectory Generation

**ECE M237: Dynamic Programming — Course Project**

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Background: Diffusion and Flow Matching for Trajectory Generation](#2-background)
3. [Background: Model-Based Diffusion (MBD)](#3-model-based-diffusion)
4. [Background: The Score Function and Its Two Sources](#4-the-score-function)
5. [The Gap: Nobody Has Merged Model-Based and Data-Driven Scores](#5-the-gap)
6. [Our Proposal: Product-of-Experts Score Fusion](#6-our-proposal)
7. [Connection to Dynamic Programming and Course Material](#7-connection-to-dp)
8. [Connection to the Instructor's Recommended Paper List](#8-connection-to-paper-list)
9. [Literature Review](#9-literature-review)
10. [Open Questions and Risks](#10-open-questions)

---

## 1. The Big Picture

In robot motion planning, we often have access to two complementary sources of information:

- **Data**: A dataset of expert demonstrations showing how to perform tasks in various environments.
- **Physics**: A dynamics model of the robot ($x_{k+1} = f(x_k, u_k)$) plus a cost function encoding the task objective.

Recent generative models (diffusion models, flow matching) have shown impressive results learning trajectory distributions from data. Separately, Model-Based Diffusion (MBD) has shown that you can compute the score function analytically from the dynamics model, without any data at all. But these two approaches have remained separate paradigms.

**Our project investigates their fusion**: can we combine the data-driven score (which captures multimodal, diverse trajectory distributions from demonstrations) with the model-based score (which guarantees dynamics feasibility and cost optimality) to get the best of both? And does this fusion have a clean connection to classical optimal control — specifically, does the model-based correction reduce to the LQR Riccati gain in the linear-quadratic case?

This is a project about **merging offline and online methods** — the central theme of ECE M237.

---

## 2. Background: Diffusion and Flow Matching for Trajectory Generation

### 2.1 Diffusion Models

A diffusion model defines a forward process that gradually adds noise to data, and a reverse process that learns to denoise. For trajectory planning, each "data sample" is an expert trajectory $\tau = (q_0, q_1, \ldots, q_H)$.

**Forward SDE** (noising):
$$d\tau_t = f_t(\tau_t)\,dt + g_t\,dW_t, \quad t \in [0, T]$$

**Reverse SDE** (denoising/generation):
$$d\tau_t = \left[f_t(\tau_t) - g_t^2 \nabla_{\tau_t} \log p_t(\tau_t)\right]dt + g_t\,d\bar{W}_t$$

The key object is the **score function** $\nabla_{\tau_t} \log p_t(\tau_t)$, which tells you "which direction to move to make the noisy trajectory look more like real data." In standard diffusion models, this score is learned from data via a neural network $s_\theta(\tau_t, t) \approx \nabla_{\tau_t} \log p_t(\tau_t)$.

**Key papers:**
- **Diffuser** (Janner et al., ICML 2022): The foundational work on diffusion for trajectory planning. Learns a joint state-action trajectory distribution and uses classifier guidance for goal-conditioning. Demonstrated on offline RL benchmarks (maze navigation, locomotion).
- **Diffusion Policy** (Chi et al., RSS 2023): Applies diffusion to action-space policy learning for robotic manipulation. Uses temporal U-Net architecture with FiLM conditioning on observations. Demonstrated on real-world manipulation tasks.
- **Motion Planning Diffusion (MPD)** (Carvalho et al., 2024): Learns a trajectory prior with diffusion models, then samples from the posterior using cost gradients during denoising. Uses B-spline parameterization for smoothness. Demonstrated on 7-DoF Panda arm.

### 2.2 Flow Matching

Flow matching (FM) is a newer alternative to diffusion that uses a deterministic ODE instead of a stochastic SDE:

$$\dot{x}_t = v_\theta(x_t, t), \quad x_0 \sim \mathcal{N}(0, I), \quad t \in [0, 1]$$

The training objective is simpler:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, (x_0, x_1)} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2$$

where $x_t = (1-t)x_0 + tx_1$ is the linear interpolation.

**Advantages over diffusion for robotics:**
- Deterministic trajectories (same noise → same trajectory)
- Faster inference (single ODE integration, no iterative denoising)
- Amenable to constraint integration (smooth ODE, compatible with optimal control theory)

**Key papers:**
- **FlowMP** (Nguyen et al., 2025): Extends FM to second-order trajectory dynamics, modeling acceleration alongside velocity. Ensures smooth, dynamically feasible robot motions. Evaluated on planar and 7-DoF environments.
- **SafeFlow** (Dai et al., 2025): Integrates FM with control barrier functions (CBFs) for safety-guaranteed robot motion planning. Achieves 100% safety rate on 2D navigation and 7-DoF manipulation. Training-free safety enforcement at test time.
- **SafeFlowMatcher** (2026): Another FM + CBF variant evaluated on maze navigation, locomotion, and block stacking.
- **UniConFlow** (Yang, Dai et al., 2026): Unified constrained FM framework enforcing both equality and inequality constraints. Evaluated on inverted pendulum, car racing, and Franka arm.
- **GoalFlow** (Xing et al., CVPR 2025): Goal-driven FM for autonomous driving trajectory generation. Single denoising step, state-of-the-art on NAVSIM benchmark.

### 2.3 The Limitation: No Dynamics Awareness

All of the above methods learn the score function (or velocity field) **entirely from data**. The neural network $s_\theta$ or $v_\theta$ has no knowledge of the robot's actual equations of motion $\dot{q} = f(q, u)$. This means:

- Generated trajectories can violate dynamics constraints (infeasible accelerations, torque limits)
- Performance degrades in regions of state space not well-covered by training data
- There is no formal connection between the learned prior and the optimal control solution

---

## 3. Model-Based Diffusion (MBD)

### 3.1 The Key Idea

Model-Based Diffusion (Pan, Yi, Shi, Qu — NeurIPS 2024) takes the opposite approach: instead of learning the score function from data, it **computes the score analytically from the dynamics model**.

Given a trajectory optimization problem:
$$\min_\tau J(\tau) \quad \text{subject to} \quad q_{k+1} = f(q_k, u_k)$$

MBD formulates the target distribution as:
$$p_0(\tau) \propto \exp(-J(\tau) / \lambda) \cdot \mathbf{1}[\tau \text{ is dynamically feasible}]$$

The score $\nabla_\tau \log p_0(\tau)$ can be estimated via **importance-weighted Monte Carlo rollouts**: sample many perturbations of the current trajectory through the dynamics model, weight them by their cost, and compute the weighted average direction.

### 3.2 Strengths and Limitations

**Strengths:**
- No training data needed — works for any robot with a known dynamics model
- Guarantees dynamic feasibility by construction
- Generalizes to new tasks, environments, and robots without retraining
- Strong performance: outperforms PPO by 59% on MuJoCo locomotion tasks

**Limitations:**
- Cannot capture multimodal trajectory distributions (no data prior)
- Requires many Monte Carlo rollouts (computationally expensive)
- MBD's paper mentions that data "can be naturally integrated" but **does not formalize this integration** — no product-of-experts decomposition, no convergence analysis, no tradeoff bound

### 3.3 The Informal Data Integration in MBD

MBD's paper (Section 4.3) briefly mentions that demonstration data can be used to improve score estimation by evaluating the likelihood of demonstrations under the model and using them to bias the Monte Carlo estimate. They show one qualitative example: using lower-body human motion capture data to make humanoid locomotion trajectories look more natural.

But this is done informally — there is no mathematical framework for how to blend the data score with the model score, no analysis of when this helps or hurts, and no connection to classical control theory.

---

## 4. The Score Function and Its Two Sources

### 4.1 Data-Driven Score

Learned from demonstrations via a neural network:
$$s_\theta(\tau_t, t) \approx \nabla_{\tau_t} \log p_t^{\text{data}}(\tau_t)$$

**What it knows:** The distribution of expert trajectories — multimodal behaviors, stylistic preferences, implicit constraints from the demonstration environment.

**What it doesn't know:** The robot's dynamics, the cost function, constraints in new environments.

### 4.2 Model-Based Score

Computed analytically from the dynamics model and cost function:
$$\nabla_{\tau_t} \log p_t^{\text{model}}(\tau_t)$$

**What it knows:** Physics, dynamics feasibility, cost optimality, new constraints.

**What it doesn't know:** The demonstration distribution, multimodal behaviors, human preferences.

### 4.3 The Natural Question

Can we combine them?

$$\nabla_{\tau_t} \log p_t^{\text{target}}(\tau_t) = \underbrace{s_\theta(\tau_t, t)}_{\text{offline, learned from data}} + \lambda \underbrace{\nabla_{\tau_t} \log p_t^{\text{model}}(\tau_t)}_{\text{online, computed from dynamics}}$$

This is a **product-of-experts** decomposition:
$$p_t^{\text{target}}(\tau_t) \propto p_t^{\text{data}}(\tau_t) \cdot \left[p_t^{\text{model}}(\tau_t)\right]^\lambda$$

The weight $\lambda$ controls the offline-online tradeoff:
- $\lambda = 0$: Pure data-driven (offline only) — diverse but potentially infeasible
- $\lambda = 1$: Equal weight — hybrid
- $\lambda \to \infty$: Pure model-based (online only) — feasible but potentially unimodal

---

## 5. The Gap: Nobody Has Merged Model-Based and Data-Driven Scores

Here is the current landscape, organized by what score source each method uses:

| Paper | Year | Score Source | Data? | Model? | Application |
|-------|------|-------------|-------|--------|-------------|
| Diffuser | 2022 | Learned (neural net) | Yes | No | Offline RL planning |
| Diffusion Policy | 2023 | Learned (neural net) | Yes | No | Manipulation |
| MPD | 2024 | Learned + cost guidance | Yes | Partial | 7-DoF manipulation |
| MBD | 2024 | Model-based (analytical) | Optional* | Yes | Trajectory optimization |
| FlowMP | 2025 | Learned (neural net) | Yes | No | Manipulation |
| SafeFlow | 2025 | Learned + CBF correction | Yes | Safety only | Safe planning |
| OC-Flow | 2025 | Learned + SOC correction | Yes | Reward only | Image/molecule generation |
| BSD | 2026 | Data-based (kernel estimation) | Yes | No | Trajectory planning |
| **Ours (proposed)** | **2026** | **Hybrid: learned + analytical** | **Yes** | **Yes** | **Robot trajectory planning** |

*MBD mentions data integration informally but does not formalize it.

**The gap is clear:** nobody has formalized the product-of-experts fusion of model-based and data-driven scores for trajectory generation, studied the $\lambda$-tradeoff, or connected the model-based correction to the Riccati equation.

The closest recent work is **Behavioral Score Diffusion (BSD)** (Li et al., April 2026), which appeared just weeks ago. BSD goes the opposite direction from MBD — it estimates scores from trajectory data via kernel methods, **without any dynamics model at all**. This further confirms that the field has split into model-only vs. data-only camps, with nobody occupying the middle ground.

---

## 6. Our Proposal: Product-of-Experts Score Fusion

### 6.1 The Algorithm

At each step of the reverse diffusion (or flow matching ODE), we use the composite score:

$$\nabla_{\tau_t} \log p_t^{\text{target}} = s_\theta(\tau_t, t) + \lambda \cdot \nabla_{\tau_t} \log p_t^{\text{model}}(\tau_t)$$

where:
- $s_\theta$ is the learned score from the offline-trained diffusion/flow matching model
- $\nabla_{\tau_t} \log p_t^{\text{model}}$ is computed online from the dynamics model (via Monte Carlo rollouts for nonlinear systems, or analytically for LQ systems)
- $\lambda \geq 0$ is the blending weight

Optionally, safety constraints can be incorporated as a third score term:

$$\nabla_{\tau_t} \log p_t^{\text{target}} = s_\theta + \lambda_{\text{model}} \cdot \nabla \log p_t^{\text{model}} + \lambda_{\text{safe}} \cdot \nabla \log p_t^{\text{safe}}$$

where $\nabla \log p_t^{\text{safe}}$ encodes collision avoidance via control barrier functions (connecting to SafeFlow's approach).

### 6.2 The LQ Special Case (Conjectured Riccati Connection)

For linear dynamics $x_{k+1} = Ax_k + Bu_k$ and quadratic cost $J = \sum_k (x_k^\top Q x_k + u_k^\top R u_k)$, we conjecture that the model-based score takes the form:

$$\nabla_{x_t} \log p_t^{\text{model}}(x_t) = -P_t \, x_t$$

where $P_t$ satisfies the **discrete-time difference Riccati equation** (Lecture 5, Eq. 23):

$$P_t = A^\top P_{t+1} A + Q - A^\top P_{t+1} B (R + B^\top P_{t+1} B)^{-1} B^\top P_{t+1} A$$

If this holds, it means the model-based correction in the generative model is **exactly the LQR optimal feedback gain** — the same object derived through dynamic programming in the course. This would be a surprising and elegant bridge between generative modeling and classical control.

### 6.3 What We Plan to Do

**Theory (best-effort):**
1. Verify the LQ-Riccati connection analytically
2. If tractable, prove convergence of the hybrid ODE in the LQ case
3. Characterize the $\lambda$-tradeoff: how does blending weight affect cost vs. feasibility?

**Implementation:**
1. Train a flow matching model on expert trajectories (2D navigation + manipulator)
2. Implement the model-based score computation (analytical for LQ, Monte Carlo for nonlinear)
3. Implement the hybrid algorithm with tunable $\lambda$
4. Run experiments: feasibility comparison, $\lambda$-sweep tradeoff curve, robustness to distribution shift, LQR ablation
5. (Optional) Add safety via CBF score term

**Fallback if theory is too hard:**
- Empirical characterization of when the hybrid helps/hurts
- The LQ-Riccati connection as the main theoretical observation
- Experimental tradeoff analysis as the main contribution

---

## 7. Connection to Dynamic Programming and Course Material

| Lecture | Topic | Connection to Our Project |
|---------|-------|--------------------------|
| **Lec 3** | System models, Lyapunov stability | The dynamics model $x_{k+1} = f(x_k, u_k)$ defines $p^{\text{model}}$. Lyapunov functions certify stability of closed-loop trajectories. |
| **Lec 4** | Bellman equation, value/policy iteration, curse of dimensionality | The model-based score $\nabla_x \log p^{\text{model}}$ is the gradient of the value function. The curse of dimensionality motivates using a learned prior. Policy iteration ≈ alternating between data score (evaluation) and model score (improvement). |
| **Lec 5** | LQR, Riccati equation, DARE, HJ-PDE | **The deepest connection.** For LQ systems, model-based score = $-P_t x$ (Riccati gain). Theorem 2 (stabilizability ↔ bounded value) governs when $p^{\text{model}}$ is well-defined. Continuous-time: model-based score satisfies the HJB PDE. |
| **Lec 6** | Exploration-exploitation | The blending weight $\lambda$ is an exploration-exploitation tradeoff: $\lambda=0$ exploits the data prior, $\lambda \to \infty$ explores via the model. |
| **Lec 7–8** | Approximate DP, function approximation, deep RL | The FM model is a function approximator for the trajectory distribution. Fitted value iteration ≈ training FM on rollout data. |
| **Lec 9–10** | MPC, rollout, MCTS, sampling-based MPC | MBD's Monte Carlo score estimation is sampling-based trajectory optimization. Adding the data prior = warm-starting MPC with an offline policy. $\lambda$ plays the role of the terminal cost in MPC. |
| **Lec 11–12** | Safety, reachability, CBFs | Safety enters as a third score term $\nabla \log p^{\text{safe}}$, connecting to CBFs and reachability analysis. |

---

## 8. Connection to the Instructor's Recommended Paper List

| Row in Instructor's List | Paper | Connection |
|--------------------------|-------|------------|
| GPU-accelerated online control & sampling-based MPC | **DIAL-MPC** (Chaoyi Pan, CMU) | Same research group as MBD. Your project formalizes what MBD left informal. Guest speaker — direct conversation opportunity. |
| Sampling-based MPC | **Monte Carlo tree search / CoVO-MPC** (also Pan's group) | MBD's Monte Carlo score estimation is sampling-based trajectory optimization in score space. |
| Integrating Generative Models | **Diffusion Guidance / CFGRL** (Frans et al.) | CFGRL's product policy $\pi \propto \hat{\pi} \cdot f(A)$ is analogous to your $p^{\text{target}} \propto p^{\text{data}} \cdot p^{\text{model}}$. Same math (product-of-experts), different domain. |
| Flow Matching / Diffusion Models | **Optimal Control Meets Flow Matching (FOCUS)** | SOC framework for guiding FM. Model-based score is a special case of SOC guidance derived from dynamics rather than a terminal reward. |
| Safety guarantee with deep neural networks | **Bellman Value Decomposition** (Sharpless) | Safety as a third score term ≈ value decomposition: total score = data + model + safety. |
| Neural Safety Certificates, Safe RL | **Safety with Agency** | CBF-based safety score connects to this row. |

---

## 9. Literature Review

### 9.1 Foundational Works: Diffusion for Planning

**Diffuser** (Janner, Du, Tenenbaum, Levine — ICML 2022). The first paper to apply diffusion models to trajectory planning. Models the joint state-action trajectory as a high-dimensional image-like object and uses a temporal U-Net for denoising. Introduces classifier guidance for goal-conditioning: at each denoising step, the score is augmented with the gradient of a goal-reaching cost. Evaluated on D4RL offline RL benchmarks. Limitation: no dynamics awareness, no safety guarantees.

**Diffusion Policy** (Chi et al. — RSS 2023). Applies diffusion to the action space rather than the full trajectory space. Uses FiLM conditioning on observations. Demonstrated on real-world manipulation tasks with impressive results. Spawned a large ecosystem of follow-up work. Limitation: action-space diffusion does not directly plan trajectories, so there is no notion of trajectory-level feasibility.

**Motion Planning Diffusion (MPD)** (Carvalho et al. — IROS 2023, updated 2024). Learns a trajectory prior with diffusion models from expert data generated by optimal planners. Uses the prior in a maximum-a-posteriori formulation with cost guidance during denoising. Introduces B-spline trajectory parameterization for smoothness. Evaluated on 2D planar and 7-DoF Panda environments with extra obstacles not seen during training. This is the closest "data-driven diffusion + cost guidance" approach to our work, but it does not use the dynamics model for score computation.

### 9.2 Model-Based Score Computation

**Model-Based Diffusion (MBD)** (Pan, Yi, Shi, Qu — NeurIPS 2024). The key paper for our project. Introduces the idea of computing the score function analytically from the dynamics model using Monte Carlo importance sampling. The target distribution is $p_0(\tau) \propto \exp(-J(\tau)/\lambda) \cdot p_d(\tau)$ where $p_d$ is a Dirac delta on dynamically feasible trajectories. The score is estimated by sampling perturbations, rolling them through the dynamics, weighting by cost, and averaging. Demonstrated on contact-rich tasks (Ant, HalfCheetah, Hopper, Humanoid) with performance comparable to RL. Mentions data integration informally but does not formalize it. **This informal mention is exactly the gap our project fills.**

**Behavioral Score Diffusion (BSD)** (Li et al. — April 2026). The newest paper in this space. Goes the opposite direction from MBD: estimates the score entirely from data using kernel-weighted Nadaraya-Watson estimation, with no dynamics model. Uses a triple-kernel weighting scheme (diffusion proximity, state context, goal relevance). Achieves 98.5% of MBD's performance without a model, using only 1000 trajectories. Proves pointwise consistency of the kernel score estimate. **BSD confirms the field has split into model-only vs. data-only — nobody is doing the hybrid.**

### 9.3 Flow Matching for Robot Planning

**FlowMP** (Nguyen et al. — IROS 2025). Extends flow matching to capture second-order trajectory dynamics (acceleration). Unlike diffusion, FM trains a continuous transformation (flow) that directly maps noise to trajectory distributions without iterative denoising. Models acceleration alongside velocity for smooth, physically executable motions. Achieves 2× faster inference than MPD with better trajectory quality. Limitation: purely data-driven, no dynamics model.

**SafeFlow** (Dai, Yang et al. — 2025). Proposes flow matching barrier functions (FMBF) that enforce safety constraints during the FM generation process. Key innovation: adapts prescribed-time control barrier functions to the flow matching setting, providing theoretical safety guarantees with probability. Evaluated on 2D navigation (100% safety rate) and 7-DoF Panda manipulation. Training-free safety enforcement at test time. Limitation: does not incorporate dynamics model for feasibility (safety ≠ dynamics feasibility).

**UniConFlow** (Yang, Dai et al. — 2026). Unified constrained FM framework for certified motion planning. Enforces both equality and inequality constraints during generation. Introduces a prescribed-time zeroing function that shapes guidance to keep trajectories close to the training distribution early in generation. Evaluated on inverted pendulum, car racing, and Franka arm. The most comprehensive constraint-handling FM method to date.

### 9.4 Guided Generation via Optimal Control

**OC-Flow** (Wang et al. — ICLR 2025). The foundational framework for training-free guided flow matching via optimal control. Formulates controlled generation as: maximize terminal reward $R(x_1)$ while penalizing control effort $\frac{1}{2}\int_0^1 \|u_t\|^2 dt$. Uses Pontryagin's Maximum Principle (PMP) to derive the co-state (adjoint) equations. Key theoretical result (Proposition 1): the running cost upper-bounds the KL divergence between guided and prior distributions. Applied to image manipulation, molecule generation, and peptide design — **not to robotics**. Open-source code available.

**FOCUS** (Bill, Simsar, Hofmann — 2025). Applies the SOC+FM framework to text-to-image generation for multi-subject fidelity. Derives a training-free test-time controller and Adjoint Matching fine-tuning. Architecture-agnostic (works on SD3.5, FLUX, SDXL). Demonstrates that SOC provides a principled way to port legacy heuristics to modern FM models. Relevance: shows that SOC guidance works well in practice, but the application is image generation, not robotics.

**Stochastic Optimal Control Matching (SOCM)** (Domingo-Enrich et al. — NeurIPS 2024). Introduces a novel IDO technique for stochastic optimal control that learns the control via least-squares regression to fit a matching vector field. Draws analogy between SOC and training continuous normalizing flows. Provides theoretical foundations for SOC-based generative modeling.

### 9.5 Hybrid and Compositional Approaches

**Unified Generation-Refinement Planning** (Mizuta et al. — 2025). Bridges guided flow matching and MPPI with bidirectional information exchange: FM provides priors for MPPI refinement, MPPI warm-starts the next FM generation. Applied to social navigation. This is the closest existing work to a "generative model + online optimization" hybrid for robotics, but it does not use the model to correct the score — it uses MPPI as a separate post-processing step.

**ScoreFusion** (2024). Fuses multiple score-based generative models via KL barycenters. Shows that pre-trained diffusion models can be combined as an ensemble. Applied to image generation. Relevant as a precedent for score composition, but does not combine model-based and data-driven scores.

**Classifier Guidance** (Dhariwal & Nichol — NeurIPS 2021). The original technique for guiding diffusion models with the gradient of a separately trained classifier. Product-of-experts at the score level: $\nabla \log p(x_t | y) = \nabla \log p(x_t) + \nabla \log p(y | x_t)$. The conceptual ancestor of our approach, but uses a learned classifier rather than an analytical model.

**CFGRL** (Frans et al. — 2025). Shows that classifier-free guidance acts as a controllable policy improvement operator: $\pi(a|s) \propto \hat{\pi}(a|s) \cdot f(A(s,a))$. The guidance weight $w$ controls the degree of improvement. Mathematically analogous to our $\lambda$-weighted product of experts, but applied to action-space policies rather than trajectory-space planning.

### 9.6 Safety in Generative Planning

**SafeDiffuser** (Xiao et al. — ICLR 2025). Applies CBFs to diffusion-based planning, ensuring generated trajectories satisfy safety constraints. Uses energy-based CBFs integrated into the denoising process. Limitation: relies on iterative SDE denoising, which is slower than FM.

**DDAT** (Bouvier et al. — 2025). Enforces dynamically admissible trajectories in diffusion policies via projection onto the feasible set. Limitation: post-hoc projection can destroy distributional structure.

**Simultaneous MRMP Diffusion (SMD)** (Liang et al. — ICLR 2025 Spotlight). Combines diffusion models with constrained optimization for multi-robot motion planning. Integrates collision avoidance and kinematic feasibility into the diffusion sampling process. Shows that compositional approaches (combining learned priors with optimization) scale to dozens of robots.

---

## 10. Open Questions and Risks

### What we're fairly confident about:
- The product-of-experts score decomposition is mathematically well-defined
- The LQ-Riccati connection can be verified analytically (it's a computation, not a conjecture about an unknown)
- The $\lambda$-sweep experiment will produce interpretable results regardless of what happens
- MBD's Monte Carlo score estimation works (NeurIPS 2024, well-tested)

### What we're uncertain about:
- Whether the hybrid ODE converges to the correct product distribution in the general (nonlinear) case — this may require regularity conditions that are hard to verify
- Whether the Monte Carlo estimate of the model-based score is accurate enough in high dimensions to actually improve over the data-driven score (it might be too noisy)
- Whether the product-of-experts assumption is appropriate — the data and model distributions may not be well-calibrated relative to each other
- How sensitive the results are to the choice of $\lambda$ — it might require per-task tuning, which would limit practical value

### Fallback plan:
If the general theory proves intractable, we frame the project around:
1. The LQ-Riccati connection (verifiable, novel, connects to course material)
2. Empirical $\lambda$-tradeoff characterization (always produces results)
3. Failure mode analysis (when does the hybrid help? when does it hurt? why?)

This is still a complete, publishable-quality course project even without the general convergence proof.
