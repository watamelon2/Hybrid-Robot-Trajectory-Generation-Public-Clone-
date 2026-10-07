# Phase 3 — Deep-dive notes

## 1. Diffusion Policy [@chi2023diffusion]

**Problem.** Standard behavior cloning struggles with multimodal action distributions and high-dimensional continuous robot actions.

**Method.** The paper frames the policy as a conditional denoising diffusion process over action sequences. At inference, actions are iteratively refined through the learned score/denoising field.

**Experiments.** The retrieved metadata reports evaluation over 12 tasks from 4 manipulation benchmarks and large average gains over prior visuomotor policy baselines.

**Limitations / open questions.** Diffusion inference is iterative and can be compute-heavy; policy quality depends heavily on observation/action representation and dataset coverage.

**Connection to this repo.** Your Square-DP result is a direct confirmation of the representation point: a Square-specific compact schema gives 66.7% standalone and 70.0% with MBD, while earlier Can-centric or retrieval settings did not solve Square.

## 2. M2Diffuser [@yan2024diffuser]

**Problem.** Mobile manipulation requires trajectory optimization in 3D scenes, where planning must reason over spatial constraints and object interactions.

**Method.** The title and retrieved metadata position M2Diffuser as diffusion-based trajectory optimization, not merely closed-loop BC.

**Experiments.** The paper appears in the retrieved database with robotics/planning tags and is highly ranked for diffusion + trajectory optimization.

**Limitations / open questions.** Need full-paper reading before claiming exact benchmark improvements or architectural details.

**Connection to this repo.** It supports framing your MBD loop as a member of the diffusion/planning family: priors provide trajectory/action geometry, while test-time optimization enforces task costs.

## 3. Guided trajectory generation with diffusion models for offline model-based optimization [@yun2024guided]

**Problem.** Offline optimization needs to generate high-value candidate trajectories without online exploration.

**Method.** The paper uses diffusion models for guided trajectory generation in an offline model-based optimization setting.

**Experiments.** The retrieved metadata tags it as diffusion + model-based optimization.

**Limitations / open questions.** The application domain is broader than robot manipulation, so it should be cited for mechanism analogy rather than direct benchmark comparison.

**Connection to this repo.** Your denoise loop with `lambda_prior` is exactly a guided-generation style computation: cost/MBD terms and prior terms are combined at test time.

## 4. Implicit Behavioral Cloning [@florence2021implicit]

**Problem.** Explicit supervised policies can average over discontinuous or multimodal expert actions, especially in contact-rich or high-dimensional tasks.

**Method.** IBC uses energy-based implicit policies instead of explicit MSE-style action regression. Retrieved metadata highlights better handling of complex, discontinuous, multi-valued functions.

**Experiments.** The abstract reports robot policy learning experiments including visual image inputs, D4RL human-expert tasks, and real-world contact-rich behaviors.

**Limitations / open questions.** Energy/implicit inference can be computationally expensive and sensitive to sampling/inference procedure.

**Connection to this repo.** This supports the central idea that action priors should be treated as energy/score geometry rather than mean actions. Your `demo_1nn` score bridge and flow-pull modes are practical score-like priors.

## 5. Behavior Retrieval [@du2023behavior]

**Problem.** Few task-specific demonstrations are valuable but insufficient; large unlabeled behavior datasets contain useful behaviors, but selecting relevant portions is nontrivial.

**Method.** The method uses small downstream expert data to query relevant behaviors from a broader unlabeled offline dataset, then trains jointly on expert and retrieved data.

**Experiments.** Retrieved metadata says the method improves few-shot imitation by balancing task-specific supervision with queried prior data.

**Limitations / open questions.** It is primarily a training-data selection method, not an inference-time control method.

**Connection to this repo.** Your project can claim a complementary axis: retrieval is used online as a nonparametric prior inside MBD rather than only as dataset augmentation.

## 6. Learning and Retrieval from Prior Data for Skill-based Imitation Learning [@nasiriany2022learning]

**Problem.** Imitation learning needs scalable use of prior data and robust generalization to novel tasks.

**Method.** The paper extracts temporally extended sensorimotor skills from prior data and uses retrieval-based data augmentation for target-task learning.

**Experiments.** The retrieved abstract reports simulated and real-world manipulation domains and improvements over prior approaches.

**Limitations / open questions.** It still routes retrieval into policy training; it does not directly test nearest-neighbor action priors as a test-time controller.

**Connection to this repo.** This is the closest retrieved neighbor for your tool_hang idea: use phase/skill structure plus retrieval, rather than a monolithic sparse objective.

## 7. Hierarchical Decision Transformer [@correia2023hierarchical]

**Problem.** Sequence policies degrade on long episodes and sparse rewards.

**Method.** HDT adds a high-level mechanism that selects subgoals for a low-level controller, replacing return-to-go with subgoal guidance.

**Experiments.** The retrieved abstract says the method was validated on OpenAI Gym, D4RL, and RoboMimic and improved many settings with varied horizons and reward frequencies.

**Limitations / open questions.** Transformer-based subgoal learning may require data scale and may not expose interpretable geometric predicates.

**Connection to this repo.** Strong support for the planned tool_hang shift: define approach/lift/align/hang subgoals and solve local MBD problems instead of increasing `H` or retuning gates.

## 8. ARCH [@sun2024arch]

**Problem.** Long-horizon robotic assembly requires both high-level sequencing and high-precision contact-rich control.

**Method.** ARCH uses a hierarchical planning framework with a low-level primitive library and a high-level policy. The abstract mentions parameterized skills such as grasping and inserting, with both RL and model-based policies.

**Experiments.** The retrieved metadata focuses on contact-rich assembly and few demonstrations for high-level policy learning.

**Limitations / open questions.** Primitive engineering may be task-specific, and the method may not preserve the simplicity of your unified driver.

**Connection to this repo.** Square/tool_hang residual failures are exactly contact-rich phase failures. ARCH suggests a practical path: keep MBD for primitive/local optimization, but add a phase scheduler.

## 9. SPIRE [@zhou2024spire]

**Problem.** Long-horizon manipulation is difficult for pure IL or pure RL; demonstrations can be limited, while RL exploration from scratch is inefficient.

**Method.** SPIRE uses TAMP to decompose tasks into smaller learning subproblems and combines imitation and reinforcement learning.

**Experiments.** The retrieved abstract reports improvements over prior approaches integrating IL, RL, and planning, including 35–50% average task-performance gains and improved demonstration efficiency.

**Limitations / open questions.** TAMP integration increases system complexity and may require symbolic predicates.

**Connection to this repo.** This supports a minimal TAMP-lite direction: do not build full symbolic planning yet, but define a few geometric phases for `tool_hang` and Square insertion residuals.

## 10. Reactive Diffusion Policy [@xue2025reactive]

**Problem.** Contact-rich manipulation needs fast reactive control and may benefit from visual-tactile feedback.

**Method.** The title and metadata describe a slow-fast visual-tactile diffusion policy for contact-rich manipulation.

**Experiments.** The API record is recent and likely preprint-level; details need full-paper validation.

**Limitations / open questions.** Tactile sensing may not exist in the current Robomimic low-dimensional setup.

**Connection to this repo.** It supports the diagnosis that Square and tool_hang failures require richer contact/alignment observability, not just stronger prior weights.

## 11. DROID [@khazatsky2024droid]

**Problem.** Robot learning datasets are often too small and too narrow for scalable generalization.

**Method.** DROID is a large-scale in-the-wild manipulation dataset.

**Experiments.** Retrieved metadata identifies it as an RSS 2024 dataset paper with high citation count.

**Limitations / open questions.** Heterogeneous large-scale data may hurt nearest-neighbor retrieval without task/family filtering.

**Connection to this repo.** DROID is relevant for a future bank-scaling extension: test whether retrieval + MBD benefits from larger real-world banks after task-family locality is understood.

## 12. Consistency Policy [@prasad2024consistency]

**Problem.** Diffusion-policy inference can be slow because it requires many denoising steps.

**Method.** The paper distills visuomotor policies using consistency-style distillation for accelerated inference.

**Experiments.** Retrieved metadata identifies it as an RSS 2024 paper.

**Limitations / open questions.** Acceleration may trade off policy quality or flexibility; full details need paper reading.

**Connection to this repo.** If learned priors become central, inference speed matters. Consistency-style priors could reduce wall time for `run_can.py` DP/MBD hybrids.
