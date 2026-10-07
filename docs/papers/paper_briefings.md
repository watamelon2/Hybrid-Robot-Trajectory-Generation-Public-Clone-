# Paper Briefings

Concise briefings for each paper in the project corpus, organized thematically. Each briefing covers the problem setting, core idea, key math, results, and relevance to our project.

---

## 1. Foundational Generative Models

### 1.1 DDPM — Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)

**Problem.** Learn a generative model $p_\theta(x_0)$ of data by reversing a discrete-time noise process.

**Core idea.** Define a forward Markov chain $q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I)$ for a variance schedule $\{\beta_t\}_{t=1}^T$. After $T$ steps, $q(x_T)\approx\mathcal{N}(0,I)$. Sampling reverses this chain with a learned Gaussian $p_\theta(x_{t-1}|x_t)=\mathcal{N}(\mu_\theta(x_t,t),\sigma_t^2 I)$.

**Key math.**
Let $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Then $q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I)$. Training minimizes the variational bound, which simplifies to the $\epsilon$-prediction loss:

$$L_{\text{simple}}=\mathbb{E}_{t,x_0,\epsilon}\bigl[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\;t)\|^2\bigr]$$

The denoiser predicts the noise $\epsilon\sim\mathcal{N}(0,I)$ that was added. The connection to score matching is $\nabla_{x_t}\log q(x_t|x_0)=-\epsilon/\sqrt{1-\bar\alpha_t}$, so the noise predictor is equivalent to a (scaled) score estimator.

**Result.** FID 3.17 on CIFAR-10 (256-step sampling), establishing diffusion models as competitive generative models.

**Relevance.** DDPM is the discrete-time backbone of Diffuser, Diffusion Policy, and the denoising framework underlying MBD's Tweedie-based score estimator.

---

### 1.2 Score SDE — Score-Based Generative Modeling through SDEs (Song et al., ICLR 2021)

**Problem.** Unify DDPM, SMLD, and other score-based models under a single continuous-time SDE framework.

**Core idea.** Noise perturbation is the forward SDE $\mathrm{d}x_t=f(t)x_t\mathrm{d}t+g(t)\mathrm{d}w_t$, with variance-preserving (VP) and variance-exploding (VE) as special cases. Sampling is the Anderson reverse-time SDE:

$$\mathrm{d}x_t=[f(t)x_t-g^2(t)\nabla_{x_t}\log p_t(x_t)]\mathrm{d}t+g(t)\mathrm{d}\bar w_t$$

where $\nabla\log p_t$ is estimated by a time-conditioned score network $s_\theta(x,t)$ trained via denoising score matching (DSM): $\min_\theta\mathbb{E}_{t,x_0,x_t}\|s_\theta(x_t,t)-\nabla_{x_t}\log q_{t|0}(x_t|x_0)\|^2$.

**Key math.**
The *probability-flow ODE* sharing the same marginals $(p_t)$:

$$\mathrm{d}x_t=\bigl[f(t)x_t-\tfrac{1}{2}g^2(t)\nabla\log p_t(x_t)\bigr]\mathrm{d}t$$

enables exact likelihood computation via the instantaneous change of variables formula. Controllable generation is achieved via the guided reverse SDE $\mathrm{d}x_t=[\ldots-g^2(t)\nabla\log p(y|x_t)]\mathrm{d}t+g(t)\mathrm{d}\bar w_t$.

**Result.** State-of-the-art on CIFAR-10 (FID ~2.20 with VE-SDE); unified theory connecting DDPM, NCSN, and probability-flow ODEs.

**Relevance.** Foundational to the entire project: the reverse SDE provides the generative backbone, and the probability-flow ODE connects diffusion to flow matching. Classifier guidance and model-based scoring both operate on the score.

---

### 1.3 Classifier Guidance — Diffusion Models Beat GANs (Dhariwal & Nichol, NeurIPS 2021)

**Problem.** Improve sample quality and enable class-conditional generation from diffusion models.

**Core idea.** Train a classifier $p_\phi(y|x_t)$ on noisy data at each noise level, then guide the reverse process by shifting the score:

$$\tilde s(x_t,t)=\nabla\log p_t(x_t)+w\cdot\nabla_{x_t}\log p_\phi(y|x_t)$$

with guidance weight $w$ controlling the quality-diversity tradeoff. Mathematically this samples from $\tilde p(x_t|y)\propto p_t(x_t)\cdot p_\phi(y|x_t)^w$.

**Key math.**
By Bayes' rule, $\nabla\log p(x|y)=\nabla\log p(x)+\nabla\log p(y|x)$. For the guided distribution at temperature $w$: $D_\text{KL}(p_w\|p_\text{data})$ decreases at the expense of diversity, formalized via $p_w(x)\propto p(x)\cdot p(y|x)^w$. They also propose *classifier-free guidance* (CFG): $\tilde\epsilon_\theta=(1+w)\epsilon_\theta(x_t,t,y)-w\epsilon_\theta(x_t,t,\varnothing)$, avoiding the need for a separate classifier.

**Result.** FID 2.97 on ImageNet 256×256, beating BigGAN.

**Relevance.** The guidance mechanism is the direct ancestor of our product-of-experts fusion: the model-based score $s_\text{mod}$ plays the role of the "classifier gradient" $\nabla\log p(y|x)$. CFG also connects to the CFGRL policy improvement perspective.

---

## 2. Flow Matching

### 2.1 Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)

**Problem.** Train a continuous normalizing flow (CNF) without costly ODE-based maximum likelihood or trace estimation.

**Core idea.** Instead of fitting a CNF by maximizing log-likelihood (requiring ODE solves during training), define a target probability path $(p_t)_{t\in[0,1]}$ from noise $p_0=\mathcal{N}(0,I)$ to data $p_1=p_\text{data}$ and regress a velocity field $v_\theta$ against the generating vector field $v_t$ satisfying the continuity equation $\partial_tp_t+\nabla\cdot(p_tv_t)=0$.

**Key math.**
The marginal flow-matching loss $\mathbb{E}_{t,x\sim p_t}\|v_\theta(x,t)-v_t(x)\|^2$ is intractable because $v_t$ depends on $p_t$. The *conditional flow matching* (CFM) trick uses per-sample paths $p_t(x|x_1)=\mathcal{N}((1-t)\cdot 0+t\cdot x_1,\sigma_t^2 I)$ with conditional velocity $v_t(x|x_1)=(x_1-x)/(1-t)$ (or simply $x_1-x_0$ for the linear interpolant $x_t=(1-t)x_0+tx_1$). The central theorem:

$$\mathbb{E}_{t,p_t}\|v_\theta-v_t\|^2 = \mathbb{E}_{t,x_1,x\sim p_t(\cdot|x_1)}\|v_\theta(x,t)-v_t(x|x_1)\|^2+C$$

where $C$ is independent of $\theta$, so minimizing the conditional loss (computable) is equivalent to minimizing the marginal loss.

**Result.** Simulation-free training of CNFs achieving competitive density estimation and generation quality with significantly fewer function evaluations than diffusion models.

**Relevance.** Flow matching is the generative backbone of our project (replacing or complementing DDPM). The CFM framework provides the training objective, and the continuity equation connects naturally to optimal transport theory.

---

## 3. Diffusion Models for Robot Trajectory Planning

### 3.1 Diffuser — Planning with Diffusion (Janner et al., ICML 2022)

**Problem.** Use generative models for flexible long-horizon planning in offline RL, avoiding the compounding-error problem of model-based rollouts.

**Core idea.** Train a DDPM over entire trajectories $\tau=(s_0,a_0,s_1,a_1,\ldots,s_H)$ from an offline dataset. At test time, condition on the start state and optionally on a goal by inpainting (clamping known values at each denoising step). Reward maximization uses *classifier guidance*: $p(\tau|y)\propto p(\tau)\cdot\mathcal{O}(\tau)^w$, where $\mathcal{O}(\tau)=\exp(\sum_t r(s_t,a_t))$ is the "optimality variable" from the control-as-inference literature.

**Key math.**
The guided denoising step modifies the mean $\mu_\theta(\tau_k,k)$:

$$\tilde\mu = \mu_\theta(\tau_k,k)+w\cdot\sigma_k^2\nabla_{\tau_k}\mathcal{J}(\mu_\theta)$$

where $\mathcal{J}$ is a learned return estimator. Inpainting enforces hard constraints by resetting known components after each denoising step.

**Result.** Matches or outperforms Decision Transformer and CQL on D4RL locomotion and Maze2D; handles goal-conditioned and reward-maximizing settings with the same model.

**Relevance.** Diffuser is the first diffusion-based trajectory planner. Our project generalizes its guidance mechanism from a learned $\nabla\mathcal{J}$ to an analytically computed $s_\text{mod}$ derived from dynamics and cost.

---

### 3.2 Diffusion Policy — Visuomotor Policy via Action Diffusion (Chi et al., RSS 2023)

**Problem.** Learn expressive, multimodal visuomotor policies for contact-rich manipulation tasks.

**Core idea.** Instead of diffusing over full state-action trajectories, Diffusion Policy generates *action chunks* $(a_t,\ldots,a_{t+H_a})$ conditioned on the current observation $o_t$ (visual or low-dimensional). It uses DDPM with a 1D temporal U-Net architecture, treating the action horizon as the "spatial" dimension.

**Key math.**
The conditional denoising process: $p_\theta(a^{k-1}_{t:t+H}|o_t)=\mathcal{N}(\mu_\theta(o_t,a^k_{t:t+H},k),\sigma_k^2I)$, with the observation $o_t$ injected via FiLM conditioning or cross-attention. Receding-horizon execution applies the first $H_e<H_a$ actions and re-plans.

**Result.** State-of-the-art on 11/12 tasks from RoboMimic and Push-T, with real-robot demonstrations on bimanual manipulation. Captures multimodal behavior (e.g., going left or right around an obstacle) that deterministic policies and VAEs miss.

**Relevance.** Demonstrates the practical power of diffusion for real robotic policy learning. Our data-driven score $s_\theta$ is trained in the same framework; we add the model-based correction on top.

---

### 3.3 MPD — Motion Planning Diffusion (Carvalho et al., IROS 2023)

**Problem.** Combine the distributional expressiveness of diffusion with cost-based guidance for robot motion planning in configuration space.

**Core idea.** Train a trajectory diffusion model on a dataset of collision-free paths, then at test time compose the score with cost-function gradients: $\tilde s(\tau_k,k)=s_\theta(\tau_k,k)+\lambda\nabla_{\tau_k}c(\hat\tau_0)$, where $\hat\tau_0$ is the Tweedie-denoised "clean" trajectory estimate and $c$ includes collision, smoothness, and goal costs. This is a practical instance of classifier guidance with a hand-crafted cost.

**Key math.**
The Tweedie estimate at step $k$: $\hat\tau_0=(\tau_k-\sqrt{1-\bar\alpha_k}\,\epsilon_\theta(\tau_k,k))/\sqrt{\bar\alpha_k}$. Guidance gradients are computed on $\hat\tau_0$ and back-propagated to $\tau_k$:

$$\nabla_{\tau_k}c(\hat\tau_0)=\frac{\partial c}{\partial\hat\tau_0}\cdot\frac{\partial\hat\tau_0}{\partial\tau_k}=\frac{1}{\sqrt{\bar\alpha_k}}\nabla_{\hat\tau_0}c(\hat\tau_0)$$

**Result.** High success rates in 7-DOF Franka planning with obstacles; outperforms sampling-based planners in cluttered environments.

**Relevance.** MPD's cost-guidance mechanism is a predecessor of our model-based score. The difference: MPD uses ad hoc cost gradients on the Tweedie estimate; we compute $s_\text{mod}$ from the full Boltzmann distribution $\exp(-J/\lambda)$ and analyze it rigorously.

---

## 4. Model-Based and Optimal-Control Approaches

### 4.1 MBD — Model-Based Diffusion (Pan et al., NeurIPS 2024)

**Problem.** Perform trajectory optimization *without any training data* by defining a score analytically from the robot's dynamics and cost function.

**Core idea.** Define a Boltzmann trajectory distribution $p_0^\text{mod}(\tau)\propto\exp(-J(\tau)/\lambda)\cdot\mathbf{1}[\tau\text{ feasible}]$, noise it via a VP-SDE to get $p_t^\text{mod}$, and compute the score $\nabla\log p_t^\text{mod}$ via Tweedie's formula: $\nabla\log p_t(\tau_t)=\frac{\sqrt{\bar\alpha_t}}{1-\bar\alpha_t}(\mathbb{E}[\tau_0|\tau_t]-\tau_t/\sqrt{\bar\alpha_t})$. The posterior mean $\mathbb{E}[\tau_0|\tau_t]$ is estimated by self-normalized importance sampling (SN-IS) over dynamically feasible rollouts:

$$\widehat{\mathbb{E}}[\tau_0|\tau_t]=\frac{\sum_i w_i\tau_0^{(i)}}{\sum_i w_i},\quad w_i=\exp(-J(\tau_0^{(i)})/\lambda)$$

where $\tau_0^{(i)}$ are sampled from the Gaussian posterior $q(\tau_0|\tau_t)=\mathcal{N}(\tau_t/\sqrt{\bar\alpha_t},(1-\bar\alpha_t)/\bar\alpha_t\cdot I)$.

**Key math.**
The score is exact in the infinite-sample limit by the consistency of SN-IS. For finite $M$, the CLT gives $\sqrt{M}(\hat s-s)\to\mathcal{N}(0,\Sigma)$, but variance can be high when $\lambda$ is small (peaked Boltzmann). The algorithm amounts to a sequence of reweighted sampling-based optimizations across denoising steps — essentially *annealed importance sampling meets MPC*.

**Result.** Solves Maze2D, Block Push, and robotic manipulation without any demonstrations; competitive with Diffuser (which requires offline data).

**Relevance.** MBD is the direct precursor of the "model-based score" in our project. We extend it by fusing $s_\text{mod}$ with a data-driven score $s_\theta$ and proving the Riccati connection in the LQ case.

---

### 4.2 SOCM — Stochastic Optimal Control Matching (Domingo-Enrich et al., NeurIPS 2024)

**Problem.** Solve high-dimensional stochastic optimal control (SOC) problems without dynamic programming or adjoint PDEs.

**Core idea.** Consider the controlled diffusion $\mathrm{d}X_t=[b(X_t)+\sigma u_t]\mathrm{d}t+\sigma\mathrm{d}W_t$ with cost $J(u)=\mathbb{E}[\int_0^T\frac{1}{2}\|u_t\|^2+\Psi(X_t)\mathrm{d}t+h(X_T)]$. By Kappen's log-transform, the optimal control $u^*(t,x)=-\sigma^\top\nabla V(t,x)$ (where $V$ is the value function) can be expressed as a conditional expectation under the *uncontrolled* diffusion, analogous to denoising score matching:

$$u^*(t,x)=\mathbb{E}^{\mathbb{Q}_{t,x}}[Z\cdot\xi_{t,T}]$$

where $Z=\exp(-h(X_T)-\int_t^T\Psi\,\mathrm{d}s)/\phi(t,x)$ is a path weight and $\xi_{t,T}$ is a stochastic integral of Brownian increments. SOCM trains $u_\theta$ by regressing against these Monte Carlo targets: $\min_\theta\mathbb{E}_{t,x}\|u_\theta(t,x)-Z\xi_{t,T}\|^2$.

**Key math.**
The key identity bridges Feynman–Kac with denoising: $\phi(t,x)=\mathbb{E}^{\mathbb{Q}_{t,x}}[\exp(-h(X_T)-\int\Psi)]$, and $\nabla\log\phi=\mathbb{E}^{\mathbb{Q}}[Z\xi]/\phi$. This is structurally identical to Vincent's denoising score matching theorem: the optimal control is a conditional expectation of a path functional, just as the optimal score is a conditional expectation of $\nabla\log p_{t|0}$.

**Result.** Solves SOC problems in up to 100 dimensions (log-normal, double-well) with better accuracy than adjoint-based methods (PMP, FBSDE).

**Relevance.** SOCM provides the formal bridge between score matching and stochastic optimal control. Its Feynman–Kac-based denoising identity justifies computing model-based scores via Monte Carlo rollouts — exactly MBD's strategy, now with rigorous foundations.

---

### 4.3 OC-Flow — Training-Free Guided Flow Matching with Optimal Control (Wang et al., ICLR 2025)

**Problem.** Guide a pre-trained flow-matching model at test time to maximize a reward $R(x_1)$ without retraining.

**Core idea.** View the flow-matching ODE $\mathrm{d}x_t=v_\theta(x_t,t)\mathrm{d}t$ as the "uncontrolled" system and add a perturbation: $\mathrm{d}x_t=[v_\theta(x_t,t)+u_t]\mathrm{d}t$. The goal is to find $u^*$ minimizing $-R(x_1)+\frac{1}{2}\int_0^1\|u_t\|^2\mathrm{d}t$. This is a deterministic optimal control problem solved via the Pontryagin Maximum Principle (PMP).

**Key math.**
The Hamiltonian is $\mathcal{H}=\frac{1}{2}\|u\|^2+\lambda^\top(v_\theta+u)$, so $u^*=-\lambda$ (the costate). The adjoint ODE is:

$$\dot\lambda_t=-\left(\frac{\partial v_\theta}{\partial x}\right)^\top\lambda_t,\quad \lambda_1=-\nabla R(x_1)$$

This is solved via a forward-backward sweep: forward the ODE to get $x_1$, compute $\nabla R(x_1)$, backward the adjoint to get $\lambda_t$, then set $u_t=-\lambda_t$. Iterate until convergence.

**Result.** Improves reward metrics on molecule generation and image generation tasks without any fine-tuning of the flow model.

**Relevance.** OC-Flow shows how PMP (Pontryagin) provides a principled guidance mechanism for flow matching. Our project uses a different (score-based) guidance mechanism, but OC-Flow's adjoint framework could serve as an alternative or complement to MBD-style sampling.

---

## 5. Flow Matching for Robotics

### 5.1 FlowMP — Conditional Flow Matching for Robot Motion (Nguyen et al., 2025)

**Problem.** Apply flow matching to robot motion planning, exploiting second-order structure (velocity and acceleration) for smoother trajectories.

**Core idea.** Train a conditional flow-matching model on demonstration trajectories represented as sequences of joint configurations. Unlike diffusion-based planners, FlowMP uses the ODE $\mathrm{d}\tau_t=v_\theta(\tau_t,t,c)\mathrm{d}t$ (no stochasticity at inference), conditioned on context $c$ (start, goal, obstacles). A key feature is the second-order formulation: the velocity field operates on a phase-space representation $(\tau,\dot\tau)$, encouraging kinematic feasibility.

**Key math.**
The CFM training loss: $\mathcal{L}=\mathbb{E}_{t,\tau_1,\tau_0\sim\mathcal{N}}\|v_\theta(\tau_t,t,c)-(\tau_1-\tau_0)\|^2$, where $\tau_t=(1-t)\tau_0+t\tau_1$ (linear interpolant). At inference, starting from $\tau_0\sim\mathcal{N}(0,I)$, a single ODE solve with an adaptive-step solver yields $\tau_1$ in 10–20 function evaluations (vs. 100+ denoising steps for DDPM-based methods).

**Result.** Faster inference (5–10×) than Diffusion Policy and MPD with comparable success rates on Franka manipulation benchmarks.

**Relevance.** Validates flow matching as a practical alternative to DDPM for robot planning. Our project uses the same CFM backbone and augments it with model-based score guidance.

---

### 5.2 GoalFlow — Goal-Driven Flow Matching for Autonomous Driving (Xing et al., CVPR 2025)

**Problem.** Generate multimodal, goal-conditioned trajectories for end-to-end autonomous driving.

**Core idea.** GoalFlow uses a conditional flow-matching model where the target $\tau_1$ is a driving trajectory and the condition includes a goal location (or set of candidate goals from a prediction module). The key design is a *one-step* flow-matching generator: the model is distilled to produce high-quality samples in a single forward pass via consistency-like training, enabling real-time deployment.

**Key math.**
Standard CFM with straight-line interpolant $\tau_t=(1-t)\tau_0+t\tau_1$, $v_t=\tau_1-\tau_0$. One-step distillation: a student $v_\theta^\text{1step}(\tau_0,c)$ is trained to directly predict $\tau_1$ from noise $\tau_0$ and context $c$, supervised by the multi-step teacher's output. Multi-goal handling: sample multiple goals from a proposal distribution and run one-step generation for each, selecting the best trajectory by a learned scoring head.

**Result.** State-of-the-art on nuScenes planning metrics with real-time inference (<50ms per trajectory).

**Relevance.** Shows flow matching scales to high-dimensional driving scenarios. The one-step distillation idea could accelerate our fused generative process at deployment time.

---

## 6. Safety and Constraints

### 6.1 SafeFlow — Flow Matching with Control Barrier Functions (Dai et al., 2025)

**Problem.** Ensure safety (obstacle avoidance) in flow-matching-based trajectory generation without sacrificing multimodality.

**Core idea.** At each ODE integration step of the flow-matching sampler, project the velocity field through a Control Barrier Function (CBF) safety filter:

$$v_\text{safe}=\arg\min_v\|v-v_\theta(\tau_t,t)\|^2\quad\text{s.t.}\quad\dot h(\tau_t,v)\ge-\alpha(h(\tau_t))$$

where $h(\tau)\ge 0$ defines the safe set and $\alpha$ is a class-$\mathcal{K}$ function. This is a pointwise QP at each solver step.

**Key math.**
The CBF condition $\dot h+\alpha(h)\ge 0$ ensures forward invariance of $\{x:h(x)\ge 0\}$. Applied to the ODE flow $\dot\tau_t=v(\tau_t,t)$, the condition becomes a linear constraint on $v$ (since $\dot h=\nabla h^\top v$), yielding a closed-form QP solution via KKT conditions.

**Result.** Zero constraint violations on 2D navigation and Franka planning with obstacles, vs. occasional violations for unconstrained flow matching and MPD-style guidance.

**Relevance.** SafeFlow demonstrates how to enforce hard safety constraints on flow-matching trajectories. This mechanism could be composed with our product-of-experts fusion to guarantee constraint satisfaction.

---

### 6.2 SafeFlowMatcher — Predictor-Corrector CBF-QP for Flow Matching (Yang et al., 2025)

**Problem.** Improve upon SafeFlow's single-step CBF projection by correcting for the discretization error in multi-step ODE solves.

**Core idea.** Uses a predictor-corrector scheme: the *predictor* takes a tentative Euler step with $v_\theta$, and the *corrector* solves a CBF-QP to pull the trajectory back into the safe set if the prediction violated the barrier. This two-phase approach handles cases where a single projection is insufficient due to large step sizes or high curvature of the safe boundary.

**Key math.**
At each step $t_i\to t_{i+1}$:
1. **Predict:** $\tilde\tau_{i+1}=\tau_i+\Delta t\cdot v_\theta(\tau_i,t_i)$
2. **Correct:** solve $\tau_{i+1}=\arg\min_\tau\|\tau-\tilde\tau_{i+1}\|^2$ s.t. $h(\tau)\ge\gamma>0$ (strengthened barrier for robustness)

This decomposes into independent per-waypoint QPs when the barrier function is separable across time steps.

**Result.** Reduces constraint violations to near-zero even with coarse ODE discretization (5 steps), while maintaining trajectory quality.

**Relevance.** Provides a refined safety mechanism compatible with any flow-matching planner. Could layer on top of our score-fusion pipeline as a post-processing safety filter.

---

### 6.3 UniConFlow — Unified Constrained Flow Matching (Yang & Dai et al., 2025)

**Problem.** Handle *multiple* heterogeneous constraints (obstacles, joint limits, kinematic feasibility) in a unified framework for flow-matching-based motion planning.

**Core idea.** Formulates the constrained flow-matching generation as a sequence of QPs that jointly enforce all constraints. Each constraint type is encoded as a differentiable barrier or equality, and the velocity field is projected at each ODE step:

$$v_\text{safe}=\arg\min_v\|v-v_\theta\|^2\quad\text{s.t.}\quad A_iv\le b_i\;\;\forall i\in\{\text{obstacle, joint, kinematic}\}$$

The key insight is unifying CBF constraints (inequality) with kinematic-feasibility constraints (equality: dynamics satisfaction) in the same QP.

**Key math.**
Each CBF contributes a half-space constraint on $v$; kinematic feasibility contributes equality constraints $x_{k+1}=f(x_k,u_k)$ linearized around the current estimate. The combined QP is solved by an off-the-shelf solver (OSQP) at each integration step, scaling linearly in the number of constraints.

**Result.** Certified constraint satisfaction across obstacle, joint-limit, and self-collision constraints simultaneously on a 7-DOF arm.

**Relevance.** UniConFlow extends the CBF approach from a single safety constraint to the full planning constraint set. If our model-based score already encodes cost/dynamics, UniConFlow's QP layer can handle the remaining hard constraints.

---

### 6.4 SafeDiffuser — Safe Planning with Diffusion Models (Xiao et al., ICLR 2025)

**Problem.** Enforce safety constraints within the DDPM denoising loop for trajectory planning.

**Core idea.** At each denoising step, after the score-based update, apply a differentiable CBF-based correction to the denoised trajectory. The "safe denoiser" replaces $\mu_\theta(\tau_k,k)$ with the projection:

$$\tilde\mu=\arg\min_\mu\|\mu-\mu_\theta\|^2\quad\text{s.t.}\quad h(\hat\tau_0(\mu))\ge 0$$

where $\hat\tau_0(\mu)$ is the Tweedie estimate derived from $\mu$ and $h$ is the barrier function. This embeds safety within the generative process rather than as a post-hoc filter.

**Key math.**
The Tweedie estimate at step $k$: $\hat\tau_0=(1/\sqrt{\bar\alpha_k})(\tau_k-(1-\bar\alpha_k)\epsilon_\theta/\sqrt{1-\bar\alpha_k})$. The CBF constraint on $\hat\tau_0$ is propagated to a constraint on the mean $\mu$ via the chain rule. A control Lyapunov function (CLF) is additionally used to ensure progress toward the goal, yielding a CBF-CLF-QP at each denoising step.

**Result.** Safe trajectories on Maze2D and robotic arm tasks with zero constraint violations, while maintaining planning performance comparable to unconstrained Diffuser.

**Relevance.** Shows how to integrate barrier-based safety directly into the score-based denoising loop. Our product-of-experts framework could absorb the CBF constraint as an additional score term: $s_\text{CBF}=\nabla\log\mathbf{1}[h(\tau)\ge 0]$.

---

### 6.5 S²Diff — Safe and Stable Control via Lyapunov-Guided Diffusion (Cheng et al., 2025)

**Problem.** Generate dynamically stable control policies using diffusion, with formal Lyapunov-based stability guarantees.

**Core idea.** Instead of learning from demonstrations, S²Diff constructs a score function from a known Lyapunov function $V(x)$ for the system $\dot x=f(x)+g(x)u$. The "stability score" is $\nabla\log p_\text{stable}(u|x)\propto\nabla_u[-V(f(x)+g(x)u)]$, which points in the direction that decreases the Lyapunov function most rapidly. This score is used either as the sole generative signal (no data) or composed with a data-driven score.

**Key math.**
The Lyapunov decrease condition $\dot V=\nabla V^\top(f+gu)\le-\alpha(V)$ is equivalent to a half-space constraint on $u$: $(\nabla V^\top g)u\le-\alpha(V)-\nabla V^\top f$. The diffusion guidance term is:

$$s_\text{Lyap}(u,x)=-\nabla_u\bigl[\nabla V(x)^\top(f(x)+g(x)u)\bigr]=-g(x)^\top\nabla V(x)$$

This is a state-dependent but *linear-in-$u$* score, analogous to our $s_\text{mod}=-\lambda^{-1}P_kx_k$ in the LQ case.

**Result.** Provably stable control on nonlinear systems (inverted pendulum, quadrotor) without any demonstration data.

**Relevance.** S²Diff is the closest relative to our model-based score concept: both derive an analytical score from system properties. The difference is that S²Diff uses a Lyapunov function (stability) while we use a cost function (optimality). In the LQ case, both reduce to the Riccati matrix.

---

## 7. Data-Driven Scores and Guidance Theory

### 7.1 BSD — Behavioral Score Diffusion: Model-Free Trajectory Planning via Kernel-Based Score Estimation (Li et al., 2026)

**Problem.** Compute a trajectory-level score function directly from data *without training a neural network*, enabling purely data-driven planning that can be composed with model-based scores.

**Core idea.** Given a dataset of trajectories $\{\tau^{(i)}\}_{i=1}^N$, BSD estimates the score $\nabla\log p_t(\tau_t)$ non-parametrically using kernel density estimation (KDE). At noise level $t$, the noised density is $p_t(\tau_t)=\frac{1}{N}\sum_i\mathcal{N}(\tau_t;\sqrt{\bar\alpha_t}\tau^{(i)},(1-\bar\alpha_t)I)$, whose score is available in closed form:

$$\nabla\log p_t(\tau_t)=\frac{\sum_i w_i(\tau_t)\bigl(\sqrt{\bar\alpha_t}\tau^{(i)}-\tau_t\bigr)}{(1-\bar\alpha_t)\sum_i w_i(\tau_t)}$$

where $w_i(\tau_t)=\mathcal{N}(\tau_t;\sqrt{\bar\alpha_t}\tau^{(i)},(1-\bar\alpha_t)I)$. This is essentially a soft nearest-neighbor regression in the noised space.

**Key math.**
The KDE score has a natural annealing property: at high noise ($t$ large), all data points contribute roughly equally (exploration); at low noise ($t$ small), the estimate is dominated by the nearest neighbor (exploitation). The connection to importance sampling is: the weights $w_i$ are the Gaussian likelihoods, and the score estimate is self-normalized importance sampling with the uniform proposal over data.

**Result.** On Maze2D and robotic manipulation, BSD matches the performance of neural-network-based Diffuser using only $O(N)$ KDE evaluations, with no training phase.

**Relevance.** BSD provides a training-free "behavioral score" $s_\text{data}$ that can be directly fused with MBD's model-based score $s_\text{mod}$ via the product-of-experts mechanism. This is the simplest realization of our score-fusion framework.

---

### 7.2 CFGRL — Diffusion Guidance Is a Controllable Policy Improvement Operator (Frans et al., 2025)

**Problem.** Provide a theoretical analysis of classifier-free guidance (CFG) as a policy improvement operator in the RL setting.

**Core idea.** Consider a diffusion-based policy $\pi_\theta(a|s)$ trained by behavior cloning. CFG with guidance weight $w>1$ effectively samples from a *tilted* policy:

$$\pi_w(a|s)\propto\pi_\theta(a|s)^{1+w}\cdot\pi_\theta(a)^{-w}\propto\pi_\theta(a|s)\cdot\left(\frac{\pi_\theta(a|s)}{\pi_\theta(a)}\right)^w$$

which upweights actions that are more likely conditioned on $s$ than marginally — i.e., actions that are most "relevant" to the current state. This is equivalent to an advantage-weighted policy improvement: $\log\pi_w\propto\log\pi_\theta+w\cdot A_\theta$, where the implicit advantage is $A_\theta(a,s)=\log\pi_\theta(a|s)-\log\pi_\theta(a)$.

**Key math.**
They prove that under mild regularity conditions, the CFG-guided policy satisfies a soft policy improvement guarantee: $\mathbb{E}_{\pi_w}[Q^{\pi_\theta}(s,a)]\ge\mathbb{E}_{\pi_\theta}[Q^{\pi_\theta}(s,a)]$ for sufficiently small $w$. The proof uses the fact that the implicit advantage has positive expected value under $\pi_w$ (Jensen's inequality on the log-ratio). For large $w$, the distribution degenerates (mode collapse), establishing an optimal $w^*$ that maximizes improvement.

**Result.** CFG with $w\approx 1\text{–}3$ improves offline RL performance across D4RL benchmarks without any additional reward training, outperforming explicit reward-guided methods.

**Relevance.** CFGRL reinterprets our guidance weight $\lambda$ as a policy-improvement knob. In our framework, $s_\text{fused}=s_\theta+\lambda\cdot s_\text{mod}$ is a weighted combination analogous to the CFG mechanism. CFGRL's analysis predicts that too-large $\lambda$ causes mode collapse — explaining the $\lambda$-sensitivity we hypothesize.
