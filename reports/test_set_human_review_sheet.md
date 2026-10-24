# RAGScaleBench — Locked Test Set Independent Human Review Sheet

**Document Purpose:** Independent human audit sheet for the 50 locked test evaluation items.
**Current Operational Status:** `unverified_programmatic` / `synthetic_template`.
**Integrity Directive:** Until verified item-by-item by a human, this set must be documented as unverified synthetic evaluation.

---

## Human Verification Rubric & Checklist

For each item below, evaluate the following four criteria:

1. **Factual Grounding (FG):** Does the ground-truth answer strictly state facts contained in the evidence span?
2. **Title Independence (TI):** Does the question rely on leaking the exact document title, or can it be understood independently?
3. **Evidence Boundary (EB):** Does the character offset correctly align with the source text without offset drift?
4. **Verdict (V):** Mark `[X] Approved`, `[ ] Rejected`, or `[ ] Revision Needed`.

---

## Evaluation Items (1 to 50)

### Item 01: `qa-test-0001` (ARXIV)
- **Document ID:** `arxiv:2610.27615v1` | **Source Locator:** [https://arxiv.org/abs/2610.27615v1](https://arxiv.org/abs/2610.27615v1)
- **Document Title:** *DISTA-LP: Privacy-Preserving Distributed Linear Programming for Metric Differential Privacy*
- **Question:** According to study 'DISTA-LP: Privacy-Preserving Distributed Linear Programming for Metric Differential Privacy', what finding or methodology is established?
- **Ground Truth Answer:** Linear-programming (LP) mechanisms for metric differential privacy (mDP) enable fine-grained, task-aware utility control, but they are computationally expensive and become privacy-fragile in distributed settings: uploading user-specific LP coefficient matrices (e.g., distance and loss blocks) to a central solver can itself leak sensitive information. We present DISTA-LP, a privacy-preserving distributed mDP optimization framework that replaces raw coefficient uploads with compact surrogate parameterizations and releases only calibrated, perturbed parameter vectors, thereby reducing both upload-stage leakage and problem dimensionality. On the server side, DISTA-LP reconstructs approximate coefficients and solves a scalable mastersubproblem formulation via Benders decomposition. To certify end-to-end privacy, we introduce a probabilistic mDP (PmDP) auditor that empirically quantifies the privacy loss induced by parameter release, allocates the required privacy budget prior to optimization, and composes it with the downstream mechanism to obtain an end-to-end mDP guarantee. Experiments on large realworld road-network datasets (Rome, New York City, and London) show that DISTA-LP achieves near-zero mDP violation rates, substantially reduces computation time compared to monolithic and hybrid baselines, and matches or improves utility, thereby demonstrating stronger upload privacy and significantly improved scalability.
- **Evidence Span (`93:1529`):** > "Linear-programming (LP) mechanisms for metric differential privacy (mDP) enable fine-grained, task-aware utility control, but they are computationally expensive and become privacy-fragile in distributed settings: uploading user-specific LP coefficient matrices (e.g., distance and loss blocks) to a central solver can itself leak sensitive information. We present DISTA-LP, a privacy-preserving distributed mDP optimization framework that replaces raw coefficient uploads with compact surrogate parameterizations and releases only calibrated, perturbed parameter vectors, thereby reducing both upload-stage leakage and problem dimensionality. On the server side, DISTA-LP reconstructs approximate coefficients and solves a scalable mastersubproblem formulation via Benders decomposition. To certify end-to-end privacy, we introduce a probabilistic mDP (PmDP) auditor that empirically quantifies the privacy loss induced by parameter release, allocates the required privacy budget prior to optimization, and composes it with the downstream mechanism to obtain an end-to-end mDP guarantee. Experiments on large realworld road-network datasets (Rome, New York City, and London) show that DISTA-LP achieves near-zero mDP violation rates, substantially reduces computation time compared to monolithic and hybrid baselines, and matches or improves utility, thereby demonstrating stronger upload privacy and significantly improved scalability."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 02: `qa-test-0002` (ARXIV)
- **Document ID:** `arxiv:2610.27616v1` | **Source Locator:** [https://arxiv.org/abs/2610.27616v1](https://arxiv.org/abs/2610.27616v1)
- **Document Title:** *EchoNav: Evidential BEV Completion from Historical Observations for Visual Navigation*
- **Question:** According to study 'EchoNav: Evidential BEV Completion from Historical Observations for Visual Navigation', what finding or methodology is established?
- **Ground Truth Answer:** Visual navigation in cluttered environments requires robots to recover previously observed structure and infer obstacles and free space that remain occluded. We present a framework that combines temporal aggregation and occupancy completion in an evidential bird's-eye-view (BEV) representation predicted from short RGB sequences. A decoder adapts frozen VGGT-Omega features and combines historical residual proposals using cellwise weights and a learned no-update option. The BEV predictor produces a local map in the latest ego frame without depth or camera-pose inputs or explicit geometric warping. The evidential model treats observed free space separately and predicts occupancy with associated confidence across the remaining supported region. We train the predictor on simulated RGB sequences paired with complete and visibility-masked occupancy annotations. Our experiments evaluate navigation in simulation and on a wheeled robot, with additional demonstrations on quadruped and humanoid robots. In simulated sequential two-goal navigation, our system achieves 89.6% success, outperforming the strongest evaluated baseline by 32.5 percentage points. Across 60 wheeled-robot trials in 20 scenes, it achieves 73.3% success, compared with 33.3% for the strongest evaluated baseline.
- **Evidence Span (`87:1376`):** > "Visual navigation in cluttered environments requires robots to recover previously observed structure and infer obstacles and free space that remain occluded. We present a framework that combines temporal aggregation and occupancy completion in an evidential bird's-eye-view (BEV) representation predicted from short RGB sequences. A decoder adapts frozen VGGT-Omega features and combines historical residual proposals using cellwise weights and a learned no-update option. The BEV predictor produces a local map in the latest ego frame without depth or camera-pose inputs or explicit geometric warping. The evidential model treats observed free space separately and predicts occupancy with associated confidence across the remaining supported region. We train the predictor on simulated RGB sequences paired with complete and visibility-masked occupancy annotations. Our experiments evaluate navigation in simulation and on a wheeled robot, with additional demonstrations on quadruped and humanoid robots. In simulated sequential two-goal navigation, our system achieves 89.6% success, outperforming the strongest evaluated baseline by 32.5 percentage points. Across 60 wheeled-robot trials in 20 scenes, it achieves 73.3% success, compared with 33.3% for the strongest evaluated baseline."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 03: `qa-test-0003` (ARXIV)
- **Document ID:** `arxiv:2610.27617v1` | **Source Locator:** [https://arxiv.org/abs/2610.27617v1](https://arxiv.org/abs/2610.27617v1)
- **Document Title:** *Provisioning Robust and Efficient LiDAR Perception for Autonomous Driving*
- **Question:** According to study 'Provisioning Robust and Efficient LiDAR Perception for Autonomous Driving', what finding or methodology is established?
- **Ground Truth Answer:** LiDAR-based 3D object detection is a core perception service for autonomous driving, but deployment requires robustness to point-cloud corruption while meeting latency and memory requirements. Existing approaches address robustness and resource efficiency separately, resulting in compact detectors that are fragile under corruption or robust detectors that remain costly to deploy. We therefore formulate LiDAR perception provisioning as constructing a detector that remains accurate under point-cloud corruptions while meeting constraints on inference latency and peak GPU memory. To achieve robustness, we first craft corrupted samples that mimic extreme weather conditions. Then we train candidate detectors offline by ordering clean and corrupted samples by perturbation sensitivity, through which detectors learn more from hard corrupted samples. We estimate perturbation sensitivity using an ensemble of seven heterogeneous 3D detectors by tracking changes in confidence, BEV overlap, and unmatched detections. To improve efficiency, we narrow the network and use offline neighborhood-density measurements to determine how many points and neighbors each sampling stage processes. We evaluate our method using AP (Average Precision) under clean and corrupted conditions, inference latency, and peak allocated GPU memory. On KITTI for Car detection with PointRCNN, our method improves Moderate AP under corruption by 3.32 points over the original detector while reducing mean latency by 19.8% and peak allocated GPU memory by 53.9%. Across two datasets, two detector architectures, and three object classes, our method improves AP under corruption by 4.03 points on average over the corresponding original detectors, while reducing mean inference latency and peak allocated GPU memory by 14.4% and 38.0%, respectively. Our code is available at https://github.com/drunklibai/robust-efficient-lidar-perception.
- **Evidence Span (`75:1988`):** > "LiDAR-based 3D object detection is a core perception service for autonomous driving, but deployment requires robustness to point-cloud corruption while meeting latency and memory requirements. Existing approaches address robustness and resource efficiency separately, resulting in compact detectors that are fragile under corruption or robust detectors that remain costly to deploy. We therefore formulate LiDAR perception provisioning as constructing a detector that remains accurate under point-cloud corruptions while meeting constraints on inference latency and peak GPU memory. To achieve robustness, we first craft corrupted samples that mimic extreme weather conditions. Then we train candidate detectors offline by ordering clean and corrupted samples by perturbation sensitivity, through which detectors learn more from hard corrupted samples. We estimate perturbation sensitivity using an ensemble of seven heterogeneous 3D detectors by tracking changes in confidence, BEV overlap, and unmatched detections. To improve efficiency, we narrow the network and use offline neighborhood-density measurements to determine how many points and neighbors each sampling stage processes. We evaluate our method using AP (Average Precision) under clean and corrupted conditions, inference latency, and peak allocated GPU memory. On KITTI for Car detection with PointRCNN, our method improves Moderate AP under corruption by 3.32 points over the original detector while reducing mean latency by 19.8% and peak allocated GPU memory by 53.9%. Across two datasets, two detector architectures, and three object classes, our method improves AP under corruption by 4.03 points on average over the corresponding original detectors, while reducing mean inference latency and peak allocated GPU memory by 14.4% and 38.0%, respectively. Our code is available at https://github.com/drunklibai/robust-efficient-lidar-perception."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 04: `qa-test-0004` (ARXIV)
- **Document ID:** `arxiv:2610.27618v1` | **Source Locator:** [https://arxiv.org/abs/2610.27618v1](https://arxiv.org/abs/2610.27618v1)
- **Document Title:** *JARVIS: From Visual Manuals to Autonomous Robotic Assembly*
- **Question:** According to study 'JARVIS: From Visual Manuals to Autonomous Robotic Assembly', what finding or methodology is established?
- **Ground Truth Answer:** Visual manuals communicate procedural knowledge through stepwise diagrams, but translating these instructions into reliable long-horizon robotic execution remains challenging. We introduce JARVIS, a method-agnostic benchmark for autonomous visual-manual-guided execution without additional language instructions. JARVIS comprises 236,679 LEGO assembly tasks across 8,987 scenes, with a carefully curated JARVIS-v1 suite of 50 tasks for interactive robotic evaluation. Tasks span varying initial structural complexities and assembly horizons, requiring agents to interpret manuals, ground actions in an evolving workspace, and execute the intended procedure. A structure-aware evaluation protocol captures geometric accuracy, structural relations, construction order, partial progress, and completion. We further introduce JARVIS-agentic, a tool-grounded embodied agent that composes robot primitives into executable programs and uses execution feedback for verification and recovery, without task-specific action trajectories or model parameter updates. Across nine multimodal and reasoning models, task-oriented primitives increase the average assembly score. Performance nevertheless declines sharply with assembly horizon, and execution errors remain difficult to correct. Prior-task experience improves initial performance, while repeated reflection provides inconsistent benefits, highlighting persistent limitations in connecting visual procedural reasoning with reliable physical execution.
- **Evidence Span (`60:1557`):** > "Visual manuals communicate procedural knowledge through stepwise diagrams, but translating these instructions into reliable long-horizon robotic execution remains challenging. We introduce JARVIS, a method-agnostic benchmark for autonomous visual-manual-guided execution without additional language instructions. JARVIS comprises 236,679 LEGO assembly tasks across 8,987 scenes, with a carefully curated JARVIS-v1 suite of 50 tasks for interactive robotic evaluation. Tasks span varying initial structural complexities and assembly horizons, requiring agents to interpret manuals, ground actions in an evolving workspace, and execute the intended procedure. A structure-aware evaluation protocol captures geometric accuracy, structural relations, construction order, partial progress, and completion. We further introduce JARVIS-agentic, a tool-grounded embodied agent that composes robot primitives into executable programs and uses execution feedback for verification and recovery, without task-specific action trajectories or model parameter updates. Across nine multimodal and reasoning models, task-oriented primitives increase the average assembly score. Performance nevertheless declines sharply with assembly horizon, and execution errors remain difficult to correct. Prior-task experience improves initial performance, while repeated reflection provides inconsistent benefits, highlighting persistent limitations in connecting visual procedural reasoning with reliable physical execution."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 05: `qa-test-0005` (ARXIV)
- **Document ID:** `arxiv:2610.27620v1` | **Source Locator:** [https://arxiv.org/abs/2610.27620v1](https://arxiv.org/abs/2610.27620v1)
- **Document Title:** *Local Allow, Global Deny: Why Per-Action Authorization Does Not Compose in LLM Agents*
- **Question:** According to study 'Local Allow, Global Deny: Why Per-Action Authorization Does Not Compose in LLM Agents', what finding or methodology is established?
- **Ground Truth Answer:** Autonomous LLM agents act through tools under per-action authorization: a local guard screens each tool call in isolation. We show that local authorization does not compose into global authorization. A compositional authorization failure (CAF) is a locally authorized execution, or a pair of locally authorized executions, that violates a global policy spanning intent, purpose, provenance, consent, and delegation. We characterize CAFs along two axes, expressiveness (single-trace safety versus relational k-safety) and observability (a ladder of monitor profiles). Our central result is an impossibility: relational purpose violations form a 2-safety hyperproperty with no per-trace locus of blame, so a per-action monitor that never raises a false alarm detects none of them, independent of the underlying model or decision procedure; they become visible only under cross-execution monitoring, which none of the six guardrails we survey implements. We make this constructive with AL-SYNTH, a SAT/SMT synthesizer (using no language model) that emits minimal CAF witnesses, each checked against an exhaustive oracle. The empirical signature is sharp: across nine LLM judges the per-action view approves the synthesized witness almost without exception, blocking it at a rate comparable to a benign control, yet the same models flag it when shown the two executions together. Per-action authorization is therefore structurally insufficient for relational policy conjuncts not reducible to single-trace safety: to bound them, a monitor must compare executions or accept false alarms.
- **Evidence Span (`87:1669`):** > "Autonomous LLM agents act through tools under per-action authorization: a local guard screens each tool call in isolation. We show that local authorization does not compose into global authorization. A compositional authorization failure (CAF) is a locally authorized execution, or a pair of locally authorized executions, that violates a global policy spanning intent, purpose, provenance, consent, and delegation. We characterize CAFs along two axes, expressiveness (single-trace safety versus relational k-safety) and observability (a ladder of monitor profiles). Our central result is an impossibility: relational purpose violations form a 2-safety hyperproperty with no per-trace locus of blame, so a per-action monitor that never raises a false alarm detects none of them, independent of the underlying model or decision procedure; they become visible only under cross-execution monitoring, which none of the six guardrails we survey implements. We make this constructive with AL-SYNTH, a SAT/SMT synthesizer (using no language model) that emits minimal CAF witnesses, each checked against an exhaustive oracle. The empirical signature is sharp: across nine LLM judges the per-action view approves the synthesized witness almost without exception, blocking it at a rate comparable to a benign control, yet the same models flag it when shown the two executions together. Per-action authorization is therefore structurally insufficient for relational policy conjuncts not reducible to single-trace safety: to bound them, a monitor must compare executions or accept false alarms."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 06: `qa-test-0006` (ARXIV)
- **Document ID:** `arxiv:2610.27621v1` | **Source Locator:** [https://arxiv.org/abs/2610.27621v1](https://arxiv.org/abs/2610.27621v1)
- **Document Title:** *A Radiation-Hardened, Fault-Aware Quantized Neural Network That Self-Heals*
- **Question:** According to study 'A Radiation-Hardened, Fault-Aware Quantized Neural Network That Self-Heals', what finding or methodology is established?
- **Ground Truth Answer:** Deep learning (DL) models are increasingly being deployed into radiation-challenging environments in spacecraft, particle accelerators, and, soon, space-based data centers. Instantiated as a learned orchestration of numbers stored in persistent memory, models are susceptible to continual degradation as high-energy particles corrupt data. At inference time, new data passed through the network fundamentally break the off-line assumptions and guarantees of a static model determined during training. To preserve the quality and efficacy of the models, fault tolerance against bit flips is a clear objective. In this work we developed a runtime method for repairing the quantized weights of neural networks (NNs) after bit flips on commodity, commercial off-the-shelf (COTS) hardware, without storing extra replicas. Our work extends the well-established Fault-Aware Training (FAT) paradigm by adding an alignment penalty to concentrate undamaged activations and weights into a low-rank subspace, and we then reconstruct flagged corrupted coordinates from the uncorrupted survivors. We call this method {\it Projection Healing} and apply it to six regression benchmarks on quantized NNs. Holding a FAT checkpoint fixed and toggling only the runtime heal gives a further ${\sim}23\%$ reduction in the native-unit excess integral. Across those datasets the method extends the operational scrub interval by a median of $2.04\times$ at a $+5\%$ RMSE degradation and $1.96\times$ at $+50\%$, compared with $1.62\times$ and $1.07\times$ for classical Single Error Correction, Double Error Detection (SECDED). This introduces a new path towards designing resilient data-driven models, complementing established hardware redundancy, ECC, and FAT-based techniques.
- **Evidence Span (`76:1831`):** > "Deep learning (DL) models are increasingly being deployed into radiation-challenging environments in spacecraft, particle accelerators, and, soon, space-based data centers. Instantiated as a learned orchestration of numbers stored in persistent memory, models are susceptible to continual degradation as high-energy particles corrupt data. At inference time, new data passed through the network fundamentally break the off-line assumptions and guarantees of a static model determined during training. To preserve the quality and efficacy of the models, fault tolerance against bit flips is a clear objective. In this work we developed a runtime method for repairing the quantized weights of neural networks (NNs) after bit flips on commodity, commercial off-the-shelf (COTS) hardware, without storing extra replicas. Our work extends the well-established Fault-Aware Training (FAT) paradigm by adding an alignment penalty to concentrate undamaged activations and weights into a low-rank subspace, and we then reconstruct flagged corrupted coordinates from the uncorrupted survivors. We call this method {\it Projection Healing} and apply it to six regression benchmarks on quantized NNs. Holding a FAT checkpoint fixed and toggling only the runtime heal gives a further ${\sim}23\%$ reduction in the native-unit excess integral. Across those datasets the method extends the operational scrub interval by a median of $2.04\times$ at a $+5\%$ RMSE degradation and $1.96\times$ at $+50\%$, compared with $1.62\times$ and $1.07\times$ for classical Single Error Correction, Double Error Detection (SECDED). This introduces a new path towards designing resilient data-driven models, complementing established hardware redundancy, ECC, and FAT-based techniques."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 07: `qa-test-0007` (ARXIV)
- **Document ID:** `arxiv:2610.27622v1` | **Source Locator:** [https://arxiv.org/abs/2610.27622v1](https://arxiv.org/abs/2610.27622v1)
- **Document Title:** *A Head Camera Is (Almost) All You Need: An Electronics-Free UMI Glove Labelled and Observed by One Moving Camera*
- **Question:** According to study 'A Head Camera Is (Almost) All You Need: An Electronics-Free UMI Glove Labelled and Observed by One Moving Camera', what finding or methodology is established?
- **Ground Truth Answer:** Handheld demonstration devices decouple data collection from the robot, but conventional designs burden the hand with a wrist camera, cables and a pose tracker (SLAM, an IMU, a VR controller, or a fixed external camera that the collector's body occludes), and burden the pipeline with as many data streams to timestamp, synchronise and cross-calibrate. We show that a single egocentric camera (USD 122) worn on the collector's head replaces the wrist camera, the pose tracker and their separate streams with one. It reads printed fiducials on an electronics-free glove (USD 22) against paper tiles taped around the workspace, giving the 6-DoF gripper pose and the jaw width from the same frames the policy will see; at deployment the same camera rides a low-cost (USD 120) arm beside the robot that mimics the collector's head from the robot's joint state. Against the robot's encoders the labels are accurate to 6.5 mm and 1.6 degrees median with 93-100% frame coverage, and a policy trained on them instead of encoders pays 3% in validation loss. Adding 194 glove demonstrations to 48 teleoperated episodes lifts rollout success from 10/25 to 17/25, level with 172 teleoperated episodes alone (18/25), and the full co-trained head-camera policy succeeds in 20/25 against 15/25 for the wrist-camera recipe, whose policies drop the object when the goal leaves their view, and 12/25 for a fixed camera.
- **Evidence Span (`114:1515`):** > "Handheld demonstration devices decouple data collection from the robot, but conventional designs burden the hand with a wrist camera, cables and a pose tracker (SLAM, an IMU, a VR controller, or a fixed external camera that the collector's body occludes), and burden the pipeline with as many data streams to timestamp, synchronise and cross-calibrate. We show that a single egocentric camera (USD 122) worn on the collector's head replaces the wrist camera, the pose tracker and their separate streams with one. It reads printed fiducials on an electronics-free glove (USD 22) against paper tiles taped around the workspace, giving the 6-DoF gripper pose and the jaw width from the same frames the policy will see; at deployment the same camera rides a low-cost (USD 120) arm beside the robot that mimics the collector's head from the robot's joint state. Against the robot's encoders the labels are accurate to 6.5 mm and 1.6 degrees median with 93-100% frame coverage, and a policy trained on them instead of encoders pays 3% in validation loss. Adding 194 glove demonstrations to 48 teleoperated episodes lifts rollout success from 10/25 to 17/25, level with 172 teleoperated episodes alone (18/25), and the full co-trained head-camera policy succeeds in 20/25 against 15/25 for the wrist-camera recipe, whose policies drop the object when the goal leaves their view, and 12/25 for a fixed camera."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 08: `qa-test-0008` (ARXIV)
- **Document ID:** `arxiv:2610.27623v1` | **Source Locator:** [https://arxiv.org/abs/2610.27623v1](https://arxiv.org/abs/2610.27623v1)
- **Document Title:** *From Outcomes to Steps: Reliable Success-Aware Critic Learning for Long-Horizon Language Agents*
- **Question:** According to study 'From Outcomes to Steps: Reliable Success-Aware Critic Learning for Long-Horizon Language Agents', what finding or methodology is established?
- **Ground Truth Answer:** Reinforcement learning for long-horizon language agents requires identifying whether intermediate actions make eventual task success more or less likely when supervision is available only at trajectory level. The main difficulty is that a critic with aggregate return-prediction accuracy may still fail to resolve the local value differences needed to assign credit to actions, especially under sparse feedback and function approximation. Conventional critics estimate future return, but policy improvement also depends on knowing whether each intermediate decision makes task success more or less likely. We introduce SCALE, an actor-critic framework that augments conventional return estimation with explicit prediction of eventual task success. At each interaction prefix, the critic predicts a return value and a Success Value, defined as the probability that the task will eventually be completed from that prefix. SCALE converts changes in Success Value into multi-step action credit through Success-GAE and modulates its contribution according to the estimated reliability of the predictor, while retaining the token-level PPO objective. Across five agent benchmarks and three model backbones, SCALE improves task performance over the compared RL baselines in most settings, with gains of up to 4.1 percentage points in success rate. Separate critic-learning analyses show that SCALE also improves critic estimation and training stability.
- **Evidence Span (`97:1543`):** > "Reinforcement learning for long-horizon language agents requires identifying whether intermediate actions make eventual task success more or less likely when supervision is available only at trajectory level. The main difficulty is that a critic with aggregate return-prediction accuracy may still fail to resolve the local value differences needed to assign credit to actions, especially under sparse feedback and function approximation. Conventional critics estimate future return, but policy improvement also depends on knowing whether each intermediate decision makes task success more or less likely. We introduce SCALE, an actor-critic framework that augments conventional return estimation with explicit prediction of eventual task success. At each interaction prefix, the critic predicts a return value and a Success Value, defined as the probability that the task will eventually be completed from that prefix. SCALE converts changes in Success Value into multi-step action credit through Success-GAE and modulates its contribution according to the estimated reliability of the predictor, while retaining the token-level PPO objective. Across five agent benchmarks and three model backbones, SCALE improves task performance over the compared RL baselines in most settings, with gains of up to 4.1 percentage points in success rate. Separate critic-learning analyses show that SCALE also improves critic estimation and training stability."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 09: `qa-test-0009` (ARXIV)
- **Document ID:** `arxiv:2610.27626v1` | **Source Locator:** [https://arxiv.org/abs/2610.27626v1](https://arxiv.org/abs/2610.27626v1)
- **Document Title:** *TriGeoBench: A Multimodal Mathematical Geometry Benchmark with Figures and TikZ*
- **Question:** According to study 'TriGeoBench: A Multimodal Mathematical Geometry Benchmark with Figures and TikZ', what finding or methodology is established?
- **Ground Truth Answer:** Multimodal Large Language Models show strong vision-language capabilities, yet their geometry-centric reasoning remains insufficiently understood. Existing mathematical benchmarks typically rely on rasterized figures and cover limited geometric knowledge areas and response formats, making it difficult to assess whether models reason over geometric structures or exploit superficial visual patterns. We introduce TriGeoBench, a geometry-centric multimodal benchmark constructed from Chinese middle- and high-school mathematics examinations. TriGeoBench contains 7,000 question-answer pairs and 12,235 geometric figures with TikZ code, spanning plane geometry, solid geometry, analytic geometry, function, and vector geometry across diverse response formats. Each instance forms a {question-answer pair, image, TikZ} triplet: question-answer pairs are stored in LaTeX, images as RGB figures, and TikZ preserves geometric topology, symbolic annotations, and relational structure. We evaluate representative open- and closed-source MLLMs through image-to-TikZ and multimodal reasoning probes. Results reveal persistent fine-grained structure and image-text alignment challenges. Oracle experiments with human-curated reference TikZ suggest that accurate executable representations provide complementary geometric cues; they do not evaluate an end-to-end conversion pipeline. TriGeoBench offers a testbed for structure-aware mathematical reasoning.
- **Evidence Span (`81:1526`):** > "Multimodal Large Language Models show strong vision-language capabilities, yet their geometry-centric reasoning remains insufficiently understood. Existing mathematical benchmarks typically rely on rasterized figures and cover limited geometric knowledge areas and response formats, making it difficult to assess whether models reason over geometric structures or exploit superficial visual patterns. We introduce TriGeoBench, a geometry-centric multimodal benchmark constructed from Chinese middle- and high-school mathematics examinations. TriGeoBench contains 7,000 question-answer pairs and 12,235 geometric figures with TikZ code, spanning plane geometry, solid geometry, analytic geometry, function, and vector geometry across diverse response formats. Each instance forms a {question-answer pair, image, TikZ} triplet: question-answer pairs are stored in LaTeX, images as RGB figures, and TikZ preserves geometric topology, symbolic annotations, and relational structure. We evaluate representative open- and closed-source MLLMs through image-to-TikZ and multimodal reasoning probes. Results reveal persistent fine-grained structure and image-text alignment challenges. Oracle experiments with human-curated reference TikZ suggest that accurate executable representations provide complementary geometric cues; they do not evaluate an end-to-end conversion pipeline. TriGeoBench offers a testbed for structure-aware mathematical reasoning."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 10: `qa-test-0010` (ARXIV)
- **Document ID:** `arxiv:2610.27628v1` | **Source Locator:** [https://arxiv.org/abs/2610.27628v1](https://arxiv.org/abs/2610.27628v1)
- **Document Title:** *PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking*
- **Question:** According to study 'PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking', what finding or methodology is established?
- **Ground Truth Answer:** Looped language models can run more iterations at test time, but additional iterations need not help them solve longer problems. For sequential state tracking, final-answer supervision does not specify how far each iteration should advance, and repeated updates can change the scale of the recurrent state. We introduce \textbf{Progress-Aligned Recurrence (PAR)}, which trains the prediction at iteration $k$ to match the state after the first $k$ operations and rescales recurrent states to their first-iteration norms. On swap-$S_5$, a Qwen2.5-0.5B model trained on at most 32 moves achieves 95.5\% mean full-state accuracy at 512 moves, or $16\times$ the training length, compared with 55.7\% reported for RLT and 0.8\% for explicit chain of thought on the same backbone. Controlled $S_5$ ablations show that removing norm anchoring reduces the extrapolation factor from $16\times$ to below $2.2\times$, and replacing progress-aligned targets with final-answer supervision reduces it to at most $8\times$. A learned completion readout selects when to answer without a supplied stopping depth. When intermediate labels are unavailable, prefix self-distillation induces the correspondence from shorter problem prefixes using only final answers. The gains apply across sequential state-tracking tasks, establishing the task structure required for depth extrapolation. These findings support training intermediate predictions and state scale together to make additional recurrent depth useful for longer problems.
- **Evidence Span (`78:1590`):** > "Looped language models can run more iterations at test time, but additional iterations need not help them solve longer problems. For sequential state tracking, final-answer supervision does not specify how far each iteration should advance, and repeated updates can change the scale of the recurrent state. We introduce \textbf{Progress-Aligned Recurrence (PAR)}, which trains the prediction at iteration $k$ to match the state after the first $k$ operations and rescales recurrent states to their first-iteration norms. On swap-$S_5$, a Qwen2.5-0.5B model trained on at most 32 moves achieves 95.5\% mean full-state accuracy at 512 moves, or $16\times$ the training length, compared with 55.7\% reported for RLT and 0.8\% for explicit chain of thought on the same backbone. Controlled $S_5$ ablations show that removing norm anchoring reduces the extrapolation factor from $16\times$ to below $2.2\times$, and replacing progress-aligned targets with final-answer supervision reduces it to at most $8\times$. A learned completion readout selects when to answer without a supplied stopping depth. When intermediate labels are unavailable, prefix self-distillation induces the correspondence from shorter problem prefixes using only final answers. The gains apply across sequential state-tracking tasks, establishing the task structure required for depth extrapolation. These findings support training intermediate predictions and state scale together to make additional recurrent depth useful for longer problems."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 11: `qa-test-0011` (ARXIV)
- **Document ID:** `arxiv:2610.27629v1` | **Source Locator:** [https://arxiv.org/abs/2610.27629v1](https://arxiv.org/abs/2610.27629v1)
- **Document Title:** *Minimum-Time Escape from the Conic Field of View of a Reeds-Shepp Pursuer: Classification and Game Value for a Slower or Equally Fast Evader*
- **Question:** According to study 'Minimum-Time Escape from the Conic Field of View of a Reeds-Shepp Pursuer: Classification and Game Value for a Slower or Equally Fast Evader', what finding or methodology is established?
- **Ground Truth Answer:** We study minimum-time escape of an omnidirectional evader from the forward conic field of view of a Reeds-Shepp pursuer with minimum turning radius $r$. For speed ratio $ν$ and cone half-angle $0<α<π/2$, we prove a global escape/holding classification: straight reversal holds the whole cone for $ν\le\sinα$, escape is confined to a barrier-bounded dart for $\sinα<ν\le1$, and escape is possible from every state for $ν>1$. For the slower or equally fast evader, $\sinα<ν\le1$, we construct the solution with Isaacs' characteristics and verify it in feedback form, using a Filippov pursuer relay and a one-sided evader branch. The finite escape time is the first root of one scalar equation, and the holding region is bounded by the nonusable parts of the rays and a barrier that closes with a convex corner on the front axis. Sample-and-hold counter-controls show that this escape time is the value of the game in nonanticipative strategies at every state except the corner. There the two strategy values differ: the pursuer's is infinite, whereas an evader that chooses which barrier branch to cross from the initial germ of the pursuer's control escapes before the barrier time $τ_I$, which is the evader's exact strategy value. An independent level-set solver, refined on fixed comparison points, agrees with the closed-form escape time and reproduces the holding region at the tested parameters.
- **Evidence Span (`142:1542`):** > "We study minimum-time escape of an omnidirectional evader from the forward conic field of view of a Reeds-Shepp pursuer with minimum turning radius $r$. For speed ratio $ν$ and cone half-angle $0<α<π/2$, we prove a global escape/holding classification: straight reversal holds the whole cone for $ν\le\sinα$, escape is confined to a barrier-bounded dart for $\sinα<ν\le1$, and escape is possible from every state for $ν>1$. For the slower or equally fast evader, $\sinα<ν\le1$, we construct the solution with Isaacs' characteristics and verify it in feedback form, using a Filippov pursuer relay and a one-sided evader branch. The finite escape time is the first root of one scalar equation, and the holding region is bounded by the nonusable parts of the rays and a barrier that closes with a convex corner on the front axis. Sample-and-hold counter-controls show that this escape time is the value of the game in nonanticipative strategies at every state except the corner. There the two strategy values differ: the pursuer's is infinite, whereas an evader that chooses which barrier branch to cross from the initial germ of the pursuer's control escapes before the barrier time $τ_I$, which is the evader's exact strategy value. An independent level-set solver, refined on fixed comparison points, agrees with the closed-form escape time and reproduces the holding region at the tested parameters."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 12: `qa-test-0012` (ARXIV)
- **Document ID:** `arxiv:2610.27631v1` | **Source Locator:** [https://arxiv.org/abs/2610.27631v1](https://arxiv.org/abs/2610.27631v1)
- **Document Title:** *A Two-Way Street: Surveying the Interplay between Code-Mixing and Native Languages in Text and Speech*
- **Question:** According to study 'A Two-Way Street: Surveying the Interplay between Code-Mixing and Native Languages in Text and Speech', what finding or methodology is established?
- **Ground Truth Answer:** Systems for code-mixed (CM) text and speech rarely start from scratch. CM data is scarce, so most work borrows from the languages being mixed, which we call native languages. This transfer can also run in reverse, when CM data is used to improve models for the native languages. Existing surveys focus on processing CM itself, and none brings the two directions together. This paper fills that gap with a systematic survey of transfer in both directions, across text and speech. We organise the surveyed work in a taxonomy of transfer direction, route and modality. Through this taxonomy, we find that most transfer runs from native languages into the CM setting. The reverse direction is younger, and in it code-switching mainly helps by aligning the two languages. Text and speech work share methods but rarely cite each other. On a shared speech benchmark, native data helps more than new architectures. These findings point to open challenges, such as how much native and CM data to mix in training, and how to evaluate CM text written in different scripts.
- **Evidence Span (`103:1164`):** > "Systems for code-mixed (CM) text and speech rarely start from scratch. CM data is scarce, so most work borrows from the languages being mixed, which we call native languages. This transfer can also run in reverse, when CM data is used to improve models for the native languages. Existing surveys focus on processing CM itself, and none brings the two directions together. This paper fills that gap with a systematic survey of transfer in both directions, across text and speech. We organise the surveyed work in a taxonomy of transfer direction, route and modality. Through this taxonomy, we find that most transfer runs from native languages into the CM setting. The reverse direction is younger, and in it code-switching mainly helps by aligning the two languages. Text and speech work share methods but rarely cite each other. On a shared speech benchmark, native data helps more than new architectures. These findings point to open challenges, such as how much native and CM data to mix in training, and how to evaluate CM text written in different scripts."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 13: `qa-test-0013` (ARXIV)
- **Document ID:** `arxiv:2610.27632v1` | **Source Locator:** [https://arxiv.org/abs/2610.27632v1](https://arxiv.org/abs/2610.27632v1)
- **Document Title:** *When Does Feedback Help? Auditing Learned Voltage-Clamp Experimental Design*
- **Question:** According to study 'When Does Feedback Help? Auditing Learned Voltage-Clamp Experimental Design', what finding or methodology is established?
- **Ground Truth Answer:** A learned scientific controller can outperform a misspecified planner without benefiting from feedback. We audit two synthetic voltage-clamp tasks with a training-selected fixed protocol, current masking, a Bayesian reference, and an excluded target. Five recurrent policies repeat one fixed protocol in all 1,200 off-grid rollouts; masking current changes no decision. In a crossed four-hypothesis task, a fixed diagnostic pulse is followed by one learned choice. Policies then use the first current, but tie a correctly specified Bayesian controller, and a fitted threshold reproduces the learned actions without being that controller. The excluded-target prediction gain is $4.6991\times10^{-6}$ nats per sample. Feedback use and protocol quality are separate tests. These cases establish neither biological discovery nor general RL superiority.
- **Evidence Span (`77:925`):** > "A learned scientific controller can outperform a misspecified planner without benefiting from feedback. We audit two synthetic voltage-clamp tasks with a training-selected fixed protocol, current masking, a Bayesian reference, and an excluded target. Five recurrent policies repeat one fixed protocol in all 1,200 off-grid rollouts; masking current changes no decision. In a crossed four-hypothesis task, a fixed diagnostic pulse is followed by one learned choice. Policies then use the first current, but tie a correctly specified Bayesian controller, and a fitted threshold reproduces the learned actions without being that controller. The excluded-target prediction gain is $4.6991\times10^{-6}$ nats per sample. Feedback use and protocol quality are separate tests. These cases establish neither biological discovery nor general RL superiority."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 14: `qa-test-0014` (ARXIV)
- **Document ID:** `arxiv:2610.27634v1` | **Source Locator:** [https://arxiv.org/abs/2610.27634v1](https://arxiv.org/abs/2610.27634v1)
- **Document Title:** *Standards-Guided and Dependency-Aware Code-Level Documentation Generation*
- **Question:** According to study 'Standards-Guided and Dependency-Aware Code-Level Documentation Generation', what finding or methodology is established?
- **Ground Truth Answer:** Automatically generating high-quality documentation for software repositories remains challenging: existing approaches largely treat documentation as unconstrained text generation, lacking explicit quality specifications, depending heavily on long-context prompts over raw code, and offering limited support for reproducible evaluation. We argue that code-level documentation generation should instead be a standards-guided software engineering task, in which documentation is generated and assessed against explicit, reusable quality requirements derived from authoritative standards. We define code documentation as natural-language descriptions of repository functions and classes, organized by source-code path. We present CODEDOC, a standards-guided and dependency-aware framework that identifies dependencies between code entities via static call-graph and AST analysis and constructs dependency-aware structured prompts combining local code context, repository paths, and documentation criteria drawn from authoritative standards and style guides. We further introduce an interactive LLM-as-a-judge protocol that supplies criterion-aligned feedback to refine generated documentation along fine-grained dimensions from formal and practice-oriented standards. Experiments on ten open-source Python repositories show that CODEDOC produces higher-quality documentation than RepoAgent and a long-context baseline under standards-aligned evaluation, with relative improvements of 11.6%-17.0% and 21.4%-29.6%, respectively, while consuming roughly an order of magnitude fewer tokens per documentation entity than the long-context baseline. A human study confirms substantial alignment between LLM-based and human judgments (Spearman's rho = 0.72), and the relative findings hold across different judge LLMs. Code is available at https://github.com/ControllableModel/CodeDoc.
- **Evidence Span (`75:1949`):** > "Automatically generating high-quality documentation for software repositories remains challenging: existing approaches largely treat documentation as unconstrained text generation, lacking explicit quality specifications, depending heavily on long-context prompts over raw code, and offering limited support for reproducible evaluation. We argue that code-level documentation generation should instead be a standards-guided software engineering task, in which documentation is generated and assessed against explicit, reusable quality requirements derived from authoritative standards. We define code documentation as natural-language descriptions of repository functions and classes, organized by source-code path. We present CODEDOC, a standards-guided and dependency-aware framework that identifies dependencies between code entities via static call-graph and AST analysis and constructs dependency-aware structured prompts combining local code context, repository paths, and documentation criteria drawn from authoritative standards and style guides. We further introduce an interactive LLM-as-a-judge protocol that supplies criterion-aligned feedback to refine generated documentation along fine-grained dimensions from formal and practice-oriented standards. Experiments on ten open-source Python repositories show that CODEDOC produces higher-quality documentation than RepoAgent and a long-context baseline under standards-aligned evaluation, with relative improvements of 11.6%-17.0% and 21.4%-29.6%, respectively, while consuming roughly an order of magnitude fewer tokens per documentation entity than the long-context baseline. A human study confirms substantial alignment between LLM-based and human judgments (Spearman's rho = 0.72), and the relative findings hold across different judge LLMs. Code is available at https://github.com/ControllableModel/CodeDoc."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 15: `qa-test-0015` (ARXIV)
- **Document ID:** `arxiv:2610.27636v1` | **Source Locator:** [https://arxiv.org/abs/2610.27636v1](https://arxiv.org/abs/2610.27636v1)
- **Document Title:** *CXLResume: Retained-State Checkpoint and Restore for In-Memory Services*
- **Question:** According to study 'CXLResume: Retained-State Checkpoint and Restore for In-Memory Services', what finding or methodology is established?
- **Ground Truth Answer:** Checkpointing and restoring an in-memory service typically require writing its memory pages to checkpoint images and refilling them at the destination. When application state already resides in shared memory that remains accessible after failover, copying those pages into and out of checkpoint images is unnecessary. We present CXLResume, a prototype that retains the service's main heap objects in shared memory while checkpointing the remaining execution state and mapping metadata. Checkpointing preserves the shared heap at the same freeze point as the serialized process state. During recovery, CXLResume remaps the same shared memory at its original virtual addresses before restoring execution, preserving direct pointers within and across shared and private memory. We evaluate CXLResume with Redis and Memcached using QEMU CXL functional emulation. At 6 GiB, CXLResume reduces combined checkpoint and restore time from 15.015 s to 0.242 s for Redis and from 19.770 s to 0.276 s for Memcached in the evaluated deployments. These results show that retaining shared application state removes large application heaps from the checkpoint data path and substantially reduces checkpoint and restore latency.
- **Evidence Span (`73:1283`):** > "Checkpointing and restoring an in-memory service typically require writing its memory pages to checkpoint images and refilling them at the destination. When application state already resides in shared memory that remains accessible after failover, copying those pages into and out of checkpoint images is unnecessary. We present CXLResume, a prototype that retains the service's main heap objects in shared memory while checkpointing the remaining execution state and mapping metadata. Checkpointing preserves the shared heap at the same freeze point as the serialized process state. During recovery, CXLResume remaps the same shared memory at its original virtual addresses before restoring execution, preserving direct pointers within and across shared and private memory. We evaluate CXLResume with Redis and Memcached using QEMU CXL functional emulation. At 6 GiB, CXLResume reduces combined checkpoint and restore time from 15.015 s to 0.242 s for Redis and from 19.770 s to 0.276 s for Memcached in the evaluated deployments. These results show that retaining shared application state removes large application heaps from the checkpoint data path and substantially reduces checkpoint and restore latency."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 16: `qa-test-0016` (ARXIV)
- **Document ID:** `arxiv:2610.27637v1` | **Source Locator:** [https://arxiv.org/abs/2610.27637v1](https://arxiv.org/abs/2610.27637v1)
- **Document Title:** *SCODEL: Shared Canonical Ontology Definitions for Food Entity Linking and Ontology-Grounded Recipe Knowledge Graphs*
- **Question:** According to study 'SCODEL: Shared Canonical Ontology Definitions for Food Entity Linking and Ontology-Grounded Recipe Knowledge Graphs', what finding or methodology is established?
- **Ground Truth Answer:** Identifying recipe entities and linking them to ontology concepts is fundamental for semantic recipe understanding. However, food mentions in recipes are often ambiguous, context-dependent, and lexically diverse, making ontology entity linking (EL) challenging. We propose SCODEL, a streamlined, model-agnostic approach based on a central idea: contextual food mentions and ontology concepts that refer to the same entity should learn the same canonical ontology-oriented definition. SCODEL uses these shared definitions as a semantic normalisation layer, providing complementary semantic information to existing EL models without requiring a new linking architecture. We further introduce ReciLinkD, the first benchmark for contextual food ontology entity linking in recipes. Experiments show that SCODEL consistently improves three modular EL models (ReFinED, GSM-EL, and BLINK): SCODEL x BLINK achieves the highest Hit@1 on seen concepts (88.33%), while SCODEL x GSM-EL achieves the highest Hit@1 on unseen concepts (83.15%), outperforming the strongest large language model (LLM) based EL baseline in each setting. Experiments on a biomedical ontology benchmark further demonstrate that SCODEL generalises beyond the food domain. Finally, we use SCODEL to ground food mentions at scale and construct an ontology-grounded recipe knowledge graph, demonstrating its utility for ontology-aware recipe retrieval and corpus-level food analysis.
- **Evidence Span (`117:1559`):** > "Identifying recipe entities and linking them to ontology concepts is fundamental for semantic recipe understanding. However, food mentions in recipes are often ambiguous, context-dependent, and lexically diverse, making ontology entity linking (EL) challenging. We propose SCODEL, a streamlined, model-agnostic approach based on a central idea: contextual food mentions and ontology concepts that refer to the same entity should learn the same canonical ontology-oriented definition. SCODEL uses these shared definitions as a semantic normalisation layer, providing complementary semantic information to existing EL models without requiring a new linking architecture. We further introduce ReciLinkD, the first benchmark for contextual food ontology entity linking in recipes. Experiments show that SCODEL consistently improves three modular EL models (ReFinED, GSM-EL, and BLINK): SCODEL x BLINK achieves the highest Hit@1 on seen concepts (88.33%), while SCODEL x GSM-EL achieves the highest Hit@1 on unseen concepts (83.15%), outperforming the strongest large language model (LLM) based EL baseline in each setting. Experiments on a biomedical ontology benchmark further demonstrate that SCODEL generalises beyond the food domain. Finally, we use SCODEL to ground food mentions at scale and construct an ontology-grounded recipe knowledge graph, demonstrating its utility for ontology-aware recipe retrieval and corpus-level food analysis."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 17: `qa-test-0017` (ARXIV)
- **Document ID:** `arxiv:2610.27638v1` | **Source Locator:** [https://arxiv.org/abs/2610.27638v1](https://arxiv.org/abs/2610.27638v1)
- **Document Title:** *Ensemble-Aware Structural Encoding (EASE): Conditioning Protein Encoders on Peptide Conformational Ensembles*
- **Question:** According to study 'Ensemble-Aware Structural Encoding (EASE): Conditioning Protein Encoders on Peptide Conformational Ensembles', what finding or methodology is established?
- **Ground Truth Answer:** Protein foundation models provide transferable representations for downstream peptide prediction tasks. However, existing representations typically reduce a peptide to a single sequence or structure, which may miss conformational variability relevant to key peptide properties, including antibacterial activity, hemolysis, toxicity, stability, and binding. For example, protease resistance correlates more strongly with conformational dynamics measured by hydrogen exchange than with average helicity. Therefore, we introduce Ensemble-Aware Structural Encoding (EASE), a framework for effectively extracting structural information from conformational ensembles for peptide representation. To extract useful information from noisy conformers, EASE selects central conformers representing a peptide's major structural states and fuses their individual representations with a permutation-invariant operator to produce a peptide-level representation for downstream prediction. We then evaluate our framework on six peptide property-prediction tasks under leakage-controlled splits. Against matched single-structure baselines, EASE improves performance consistently on six tasks with a mean absolute gain of about 3.0% across two structural encoders. Gains keep growing as we add conformers through K=20. A single peptide conformer therefore provides a limited basis for structural representation, whereas our framework distilling and fusing multiple conformers yields a more informative representation.
- **Evidence Span (`110:1608`):** > "Protein foundation models provide transferable representations for downstream peptide prediction tasks. However, existing representations typically reduce a peptide to a single sequence or structure, which may miss conformational variability relevant to key peptide properties, including antibacterial activity, hemolysis, toxicity, stability, and binding. For example, protease resistance correlates more strongly with conformational dynamics measured by hydrogen exchange than with average helicity. Therefore, we introduce Ensemble-Aware Structural Encoding (EASE), a framework for effectively extracting structural information from conformational ensembles for peptide representation. To extract useful information from noisy conformers, EASE selects central conformers representing a peptide's major structural states and fuses their individual representations with a permutation-invariant operator to produce a peptide-level representation for downstream prediction. We then evaluate our framework on six peptide property-prediction tasks under leakage-controlled splits. Against matched single-structure baselines, EASE improves performance consistently on six tasks with a mean absolute gain of about 3.0% across two structural encoders. Gains keep growing as we add conformers through K=20. A single peptide conformer therefore provides a limited basis for structural representation, whereas our framework distilling and fusing multiple conformers yields a more informative representation."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 18: `qa-test-0018` (ARXIV)
- **Document ID:** `arxiv:2610.27641v1` | **Source Locator:** [https://arxiv.org/abs/2610.27641v1](https://arxiv.org/abs/2610.27641v1)
- **Document Title:** *BCT: Bayes-Consistent Tempering for Flow Language Models*
- **Question:** According to study 'BCT: Bayes-Consistent Tempering for Flow Language Models', what finding or methodology is established?
- **Ground Truth Answer:** Temperature scaling is typically applied as a post-hoc transformation of model probabilities. In posterior-induced flow language models, this creates two mismatches. First, posterior tempering implicitly changes both the endpoint distribution and the distribution of intermediate noisy states, while the denoiser remains trained on the original states. Second, the tempered posterior weights are combined with transport dynamics derived for the untempered flow. These mismatches are reflected empirically: lowering temperature improves GSM8K performance much less reliably in flow LMs than in discrete diffusion LMs. We introduce Bayes-Consistent Tempering (BCT), which makes temperature explicit in both the distributions and dynamics of the generative flow. Its learned realization trains the denoiser on tempered intermediate states and tempers the endpoint distribution using the corresponding tempered dynamics. We also derive training-free variants for conventionally trained checkpoints. Across hyperspherical flow LMs with two denoiser architectures and a von Mises--Fisher flow LM, BCT improves GSM8K generation across broad temperature ranges and achieves the strongest tempering performance. The gains also extend to Sudoku, demonstrating benefits on tightly constrained generation. At very low temperatures, the learned realization is also markedly more robust than the training-free variants, supporting temperature as a property of the generative flow across training and inference rather than a post-hoc probability adjustment. Code and model weights are available at https://daioba.github.io/bct .
- **Evidence Span (`58:1671`):** > "Temperature scaling is typically applied as a post-hoc transformation of model probabilities. In posterior-induced flow language models, this creates two mismatches. First, posterior tempering implicitly changes both the endpoint distribution and the distribution of intermediate noisy states, while the denoiser remains trained on the original states. Second, the tempered posterior weights are combined with transport dynamics derived for the untempered flow. These mismatches are reflected empirically: lowering temperature improves GSM8K performance much less reliably in flow LMs than in discrete diffusion LMs. We introduce Bayes-Consistent Tempering (BCT), which makes temperature explicit in both the distributions and dynamics of the generative flow. Its learned realization trains the denoiser on tempered intermediate states and tempers the endpoint distribution using the corresponding tempered dynamics. We also derive training-free variants for conventionally trained checkpoints. Across hyperspherical flow LMs with two denoiser architectures and a von Mises--Fisher flow LM, BCT improves GSM8K generation across broad temperature ranges and achieves the strongest tempering performance. The gains also extend to Sudoku, demonstrating benefits on tightly constrained generation. At very low temperatures, the learned realization is also markedly more robust than the training-free variants, supporting temperature as a property of the generative flow across training and inference rather than a post-hoc probability adjustment. Code and model weights are available at https://daioba.github.io/bct ."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 19: `qa-test-0019` (ARXIV)
- **Document ID:** `arxiv:2610.27644v1` | **Source Locator:** [https://arxiv.org/abs/2610.27644v1](https://arxiv.org/abs/2610.27644v1)
- **Document Title:** *Secure Multimodal Federated Optimization via Hierarchical DNA-Based Key Management*
- **Question:** According to study 'Secure Multimodal Federated Optimization via Hierarchical DNA-Based Key Management', what finding or methodology is established?
- **Ground Truth Answer:** Federated optimization over heterogeneous multimodal data enables distributed institutions to collaboratively optimize a shared model using medical images, genomic profiles, clinical records, and physiological signals without centralizing raw data. However, protecting modality-specific data throughout the optimization lifecycle remains insufficiently explored, particularly when heterogeneous modalities require separate encryption keys and selective local access. This paper proposes a secure multimodal federated optimization framework that separates local data protection from federated-update protection. A trusted key-management authority securely provisions one client key through Hybrid Public Key Encryption (HPKE), from which each client locally derives modality-specific symmetric keys using the domain-separated HMAC-based Extract-and-Expand Key Derivation Function (HKDF). Heterogeneous modalities are mapped into a unified DNA-domain representation and processed through a DNA-origami-inspired structural transformation before authenticated encryption with AES-GCM. During local optimization, only the required modalities are decrypted within an authorized client environment. Individual model updates are independently protected through secure aggregation, allowing the coordinator to observe only aggregated updates rather than each client's contribution. Together, these components establish a structured approach to modality-specific protection and secure collaborative optimization, integrating hierarchical key management, authenticated local encryption, and aggregation-level update protection within a unified framework.
- **Evidence Span (`84:1727`):** > "Federated optimization over heterogeneous multimodal data enables distributed institutions to collaboratively optimize a shared model using medical images, genomic profiles, clinical records, and physiological signals without centralizing raw data. However, protecting modality-specific data throughout the optimization lifecycle remains insufficiently explored, particularly when heterogeneous modalities require separate encryption keys and selective local access. This paper proposes a secure multimodal federated optimization framework that separates local data protection from federated-update protection. A trusted key-management authority securely provisions one client key through Hybrid Public Key Encryption (HPKE), from which each client locally derives modality-specific symmetric keys using the domain-separated HMAC-based Extract-and-Expand Key Derivation Function (HKDF). Heterogeneous modalities are mapped into a unified DNA-domain representation and processed through a DNA-origami-inspired structural transformation before authenticated encryption with AES-GCM. During local optimization, only the required modalities are decrypted within an authorized client environment. Individual model updates are independently protected through secure aggregation, allowing the coordinator to observe only aggregated updates rather than each client's contribution. Together, these components establish a structured approach to modality-specific protection and secure collaborative optimization, integrating hierarchical key management, authenticated local encryption, and aggregation-level update protection within a unified framework."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 20: `qa-test-0020` (ARXIV)
- **Document ID:** `arxiv:2610.27645v1` | **Source Locator:** [https://arxiv.org/abs/2610.27645v1](https://arxiv.org/abs/2610.27645v1)
- **Document Title:** *CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents*
- **Question:** According to study 'CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents', what finding or methodology is established?
- **Ground Truth Answer:** As computer-use agents (CUAs) move to long-horizon workflows spanning multiple applications, undetected errors can silently corrupt persistent state and cause cascading failures in subsequent actions. Reliable error detection and recovery have become critical bottlenecks for building robust CUAs. However, existing benchmarks focus primarily on end-to-end task performance, providing limited insight into whether agents can detect and recover from errors during execution. The few benchmarks that evaluate recovery rely largely on manually injected errors, which may not reflect the types or horizons of failures induced by real agent policies. We introduce CUA-Recovery, a benchmark comprising test and training suites across multiple desktop applications. We construct it using ReRail, a framework for synthesizing long-horizon tasks and trajectories in a persistent desktop environment. ReRail composes MyPCBench tasks into new long-horizon workflows with executable ground truth, collects verifier-confirmed failures from multiple policies, and turns them into verified recovery trajectories. CUA-Recovery comprises a human-verified test suite of 2,250 erroneous states over 100 tasks and a training suite of 8.6K trajectories over 377 tasks. Evaluation on the test suite shows that agents struggle both to recognize inherited errors and to complete the task. Across six agents and six takeover depths, Error Awareness Rate (EAR) reaches at most 44.0% and Pass@3 at most 25.5%. After fine-tuning Qwen3.5-35B-A3B on the training suite, its average EAR rises from 19.9% to 35.2% on held-out recovery states, and its Rubric Score goes from 5.0% to 16.1%. It also gains 15.3 points on MyPCBench and 10.8 points on Odysseys for Pass@3, showing that the training trajectories help both with recovery and with completing tasks from a clean start. Our code is available at https://github.com/xu-hu-2002/CUA-Recovery.
- **Evidence Span (`79:1992`):** > "As computer-use agents (CUAs) move to long-horizon workflows spanning multiple applications, undetected errors can silently corrupt persistent state and cause cascading failures in subsequent actions. Reliable error detection and recovery have become critical bottlenecks for building robust CUAs. However, existing benchmarks focus primarily on end-to-end task performance, providing limited insight into whether agents can detect and recover from errors during execution. The few benchmarks that evaluate recovery rely largely on manually injected errors, which may not reflect the types or horizons of failures induced by real agent policies. We introduce CUA-Recovery, a benchmark comprising test and training suites across multiple desktop applications. We construct it using ReRail, a framework for synthesizing long-horizon tasks and trajectories in a persistent desktop environment. ReRail composes MyPCBench tasks into new long-horizon workflows with executable ground truth, collects verifier-confirmed failures from multiple policies, and turns them into verified recovery trajectories. CUA-Recovery comprises a human-verified test suite of 2,250 erroneous states over 100 tasks and a training suite of 8.6K trajectories over 377 tasks. Evaluation on the test suite shows that agents struggle both to recognize inherited errors and to complete the task. Across six agents and six takeover depths, Error Awareness Rate (EAR) reaches at most 44.0% and Pass@3 at most 25.5%. After fine-tuning Qwen3.5-35B-A3B on the training suite, its average EAR rises from 19.9% to 35.2% on held-out recovery states, and its Rubric Score goes from 5.0% to 16.1%. It also gains 15.3 points on MyPCBench and 10.8 points on Odysseys for Pass@3, showing that the training trajectories help both with recovery and with completing tasks from a clean start. Our code is available at https://github.com/xu-hu-2002/CUA-Recovery."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 21: `qa-test-0021` (WIKIPEDIA)
- **Document ID:** `wikipedia:12782525` | **Source Locator:** [https://en.wikipedia.org/?curid=12782525](https://en.wikipedia.org/?curid=12782525)
- **Document Title:** *Toronto Harbour Light*
- **Question:** In the context of 'Toronto Harbour Light', what factual principle is described?
- **Ground Truth Answer:** The Toronto Harbour Light is an automated lighthouse at Vicki Keith Point on the Leslie Street Spit in Toronto, Ontario, Canada.
- **Evidence Span (`23:151`):** > "The Toronto Harbour Light is an automated lighthouse at Vicki Keith Point on the Leslie Street Spit in Toronto, Ontario, Canada."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 22: `qa-test-0022` (WIKIPEDIA)
- **Document ID:** `wikipedia:14593032` | **Source Locator:** [https://en.wikipedia.org/?curid=14593032](https://en.wikipedia.org/?curid=14593032)
- **Document Title:** *Backward inhibition*
- **Question:** In the context of 'Backward inhibition', what factual principle is described?
- **Ground Truth Answer:** In experimental psychology, backward inhibition, is a theory of sequential task control asserting that switching between tasks requires the just-completed task to be suppressed to allow a new task to be completed.  Support for the theory comes from research which has observed larger response times when returning to a task after an intermediate task than when completing three, or more, different tasks in a row. This typically comes in an ABA format, with the response time of task A the second time taking longer after having completed task B. Backward inhibition is not seen in scenarios with an ABC format, where no task is being repeated.
- **Evidence Span (`21:665`):** > "In experimental psychology, backward inhibition, is a theory of sequential task control asserting that switching between tasks requires the just-completed task to be suppressed to allow a new task to be completed.  Support for the theory comes from research which has observed larger response times when returning to a task after an intermediate task than when completing three, or more, different tasks in a row. This typically comes in an ABA format, with the response time of task A the second time taking longer after having completed task B. Backward inhibition is not seen in scenarios with an ABC format, where no task is being repeated."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 23: `qa-test-0023` (WIKIPEDIA)
- **Document ID:** `wikipedia:16923850` | **Source Locator:** [https://en.wikipedia.org/?curid=16923850](https://en.wikipedia.org/?curid=16923850)
- **Document Title:** *Swiss 1. Liga*
- **Question:** In the context of 'Swiss 1. Liga', what factual principle is described?
- **Ground Truth Answer:** Swiss Promotion League or Swiss 1. Liga Promotion, third tier of the Swiss football league system
- **Evidence Span (`196:293`):** > "Swiss Promotion League or Swiss 1. Liga Promotion, third tier of the Swiss football league system"
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 24: `qa-test-0024` (WIKIPEDIA)
- **Document ID:** `wikipedia:17984055` | **Source Locator:** [https://en.wikipedia.org/?curid=17984055](https://en.wikipedia.org/?curid=17984055)
- **Document Title:** *Koulniéré*
- **Question:** In the context of 'Koulniéré', what factual principle is described?
- **Ground Truth Answer:** Koulniéré is a village in the Tikare Department of Bam Province in northern Burkina Faso.
- **Evidence Span (`11:100`):** > "Koulniéré is a village in the Tikare Department of Bam Province in northern Burkina Faso."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 25: `qa-test-0025` (WIKIPEDIA)
- **Document ID:** `wikipedia:29799271` | **Source Locator:** [https://en.wikipedia.org/?curid=29799271](https://en.wikipedia.org/?curid=29799271)
- **Document Title:** *Amma Rajinama*
- **Question:** In the context of 'Amma Rajinama', what factual principle is described?
- **Ground Truth Answer:** It is a musical film with a film score composed by K. Chakravarthy. "Evaru Rayagalaru Amma Anu Maatakanna Kammani Kavyam" and "Srushitikarta Oka Brahmma…Atanini Srushitinchindoka Amma" are some of the best lyrics written about the greatness of the Mother. This movie also marks the debut of cinematographer Chota K. Naidu, who was an associate and regarded as a student of Dasari Narayana Rao. The movie was remade in Kannada as Amma (2001) starring Lakshmi in the title role.
- **Evidence Span (`295:771`):** > "It is a musical film with a film score composed by K. Chakravarthy. "Evaru Rayagalaru Amma Anu Maatakanna Kammani Kavyam" and "Srushitikarta Oka Brahmma…Atanini Srushitinchindoka Amma" are some of the best lyrics written about the greatness of the Mother. This movie also marks the debut of cinematographer Chota K. Naidu, who was an associate and regarded as a student of Dasari Narayana Rao. The movie was remade in Kannada as Amma (2001) starring Lakshmi in the title role."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 26: `qa-test-0026` (WIKIPEDIA)
- **Document ID:** `wikipedia:34054537` | **Source Locator:** [https://en.wikipedia.org/?curid=34054537](https://en.wikipedia.org/?curid=34054537)
- **Document Title:** *Scotura transversa*
- **Question:** In the context of 'Scotura transversa', what factual principle is described?
- **Ground Truth Answer:** Scotura transversa is a moth of the family Notodontidae. It is found in French Guiana, Suriname and Guyana.
- **Evidence Span (`20:127`):** > "Scotura transversa is a moth of the family Notodontidae. It is found in French Guiana, Suriname and Guyana."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 27: `qa-test-0027` (WIKIPEDIA)
- **Document ID:** `wikipedia:35434477` | **Source Locator:** [https://en.wikipedia.org/?curid=35434477](https://en.wikipedia.org/?curid=35434477)
- **Document Title:** *List of songs recorded by 4Minute*
- **Question:** In the context of 'List of songs recorded by 4Minute', what factual principle is described?
- **Ground Truth Answer:** Here is a complete list of songs by the Korean girl group 4Minute.
- **Evidence Span (`35:101`):** > "Here is a complete list of songs by the Korean girl group 4Minute."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 28: `qa-test-0028` (WIKIPEDIA)
- **Document ID:** `wikipedia:3951130` | **Source Locator:** [https://en.wikipedia.org/?curid=3951130](https://en.wikipedia.org/?curid=3951130)
- **Document Title:** *Spying on the United Nations*
- **Question:** In the context of 'Spying on the United Nations', what factual principle is described?
- **Ground Truth Answer:** In one notable incident, the US and other Western countries were found to be spying on the UN in March 2003, in the run-up to the Iraq War, and actual bugging devices were found inside the UN. In 2010, leaked diplomatic cables revealed that US diplomats had been spying on UN leaders. On 25 August 2013, Der Spiegel revealed US National Security Agency secretly monitored the U.N.'s internal video conferencing system via decryption in 2012.
- **Evidence Span (`421:862`):** > "In one notable incident, the US and other Western countries were found to be spying on the UN in March 2003, in the run-up to the Iraq War, and actual bugging devices were found inside the UN. In 2010, leaked diplomatic cables revealed that US diplomats had been spying on UN leaders. On 25 August 2013, Der Spiegel revealed US National Security Agency secretly monitored the U.N.'s internal video conferencing system via decryption in 2012."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 29: `qa-test-0029` (WIKIPEDIA)
- **Document ID:** `wikipedia:43010182` | **Source Locator:** [https://en.wikipedia.org/?curid=43010182](https://en.wikipedia.org/?curid=43010182)
- **Document Title:** *1895 in basketball*
- **Question:** In the context of '1895 in basketball', what factual principle is described?
- **Ground Truth Answer:** The following are the basketball events of the year 1895  throughout the world.
- **Evidence Span (`20:99`):** > "The following are the basketball events of the year 1895  throughout the world."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 30: `qa-test-0030` (WIKIPEDIA)
- **Document ID:** `wikipedia:43266412` | **Source Locator:** [https://en.wikipedia.org/?curid=43266412](https://en.wikipedia.org/?curid=43266412)
- **Document Title:** *Sharknado: The Video Game*
- **Question:** In the context of 'Sharknado: The Video Game', what factual principle is described?
- **Ground Truth Answer:** Sharknado: The Video Game is a 2014 endless running video game developed by Other Ocean Interactive and published by Majesco. The game is based on the 2014 film Sharknado 2: The Second One.
- **Evidence Span (`27:216`):** > "Sharknado: The Video Game is a 2014 endless running video game developed by Other Ocean Interactive and published by Majesco. The game is based on the 2014 film Sharknado 2: The Second One."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 31: `qa-test-0031` (WIKIPEDIA)
- **Document ID:** `wikipedia:5177338` | **Source Locator:** [https://en.wikipedia.org/?curid=5177338](https://en.wikipedia.org/?curid=5177338)
- **Document Title:** *Whitford Development Scheme*
- **Question:** In the context of 'Whitford Development Scheme', what factual principle is described?
- **Ground Truth Answer:** The 'centre' of the Whitford area could be seen as the main road intersection of the east–west Whitfords Avenue (Route 83) and the north–south Marmion Avenue (Route 71). Many businesses and locations (shops, schools, churches etc.) in the surrounding suburbs use either 'Whitford' or 'Whitfords' in its name. Some examples include Westfield Whitford City (formerly Whitford City Shopping Centre), Whitford Catholic Primary School, Whitford Church, and Whitfords railway station.
- **Evidence Span (`399:877`):** > "The 'centre' of the Whitford area could be seen as the main road intersection of the east–west Whitfords Avenue (Route 83) and the north–south Marmion Avenue (Route 71). Many businesses and locations (shops, schools, churches etc.) in the surrounding suburbs use either 'Whitford' or 'Whitfords' in its name. Some examples include Westfield Whitford City (formerly Whitford City Shopping Centre), Whitford Catholic Primary School, Whitford Church, and Whitfords railway station."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 32: `qa-test-0032` (WIKIPEDIA)
- **Document ID:** `wikipedia:52832565` | **Source Locator:** [https://en.wikipedia.org/?curid=52832565](https://en.wikipedia.org/?curid=52832565)
- **Document Title:** *Sébastien Loubsens*
- **Question:** In the context of 'Sébastien Loubsens', what factual principle is described?
- **Ground Truth Answer:** Sébastien Loubsens (born 15 February 1973) is a French-born former Spanish rugby union player. He played as a centre.
- **Evidence Span (`20:137`):** > "Sébastien Loubsens (born 15 February 1973) is a French-born former Spanish rugby union player. He played as a centre."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 33: `qa-test-0033` (WIKIPEDIA)
- **Document ID:** `wikipedia:54083732` | **Source Locator:** [https://en.wikipedia.org/?curid=54083732](https://en.wikipedia.org/?curid=54083732)
- **Document Title:** *Provincial Disaster Management Authority (Khyber Pakhtunkhwa)*
- **Question:** In the context of 'Provincial Disaster Management Authority (Khyber Pakhtunkhwa)', what factual principle is described?
- **Ground Truth Answer:** In May 2019, the Fata Disaster Management Authority (FDMA) was abolished and its functions were transferred to Provincial Disaster Management Authority (Khyber Pakhtunkhwa), after the Federally Administered Tribal Areas were merged into Khyber Pakhtunkhwa after the passing of the Twenty-fifth and Twenty-sixth Amendments to the Constitution of Pakistan.
- **Evidence Span (`430:784`):** > "In May 2019, the Fata Disaster Management Authority (FDMA) was abolished and its functions were transferred to Provincial Disaster Management Authority (Khyber Pakhtunkhwa), after the Federally Administered Tribal Areas were merged into Khyber Pakhtunkhwa after the passing of the Twenty-fifth and Twenty-sixth Amendments to the Constitution of Pakistan."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 34: `qa-test-0034` (WIKIPEDIA)
- **Document ID:** `wikipedia:54541095` | **Source Locator:** [https://en.wikipedia.org/?curid=54541095](https://en.wikipedia.org/?curid=54541095)
- **Document Title:** *2017–18 Championnat National 2*
- **Question:** In the context of '2017–18 Championnat National 2', what factual principle is described?
- **Ground Truth Answer:** The 2017–18 Championnat National 2 was the 20th season of the fourth tier in the French football league system. The competition was contested by 64 clubs split geographically across 4 groups of 16 teams each. The teams included amateur clubs (although a few were semi-professional) and the reserve teams of professional clubs.
- **Evidence Span (`32:358`):** > "The 2017–18 Championnat National 2 was the 20th season of the fourth tier in the French football league system. The competition was contested by 64 clubs split geographically across 4 groups of 16 teams each. The teams included amateur clubs (although a few were semi-professional) and the reserve teams of professional clubs."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 35: `qa-test-0035` (WIKIPEDIA)
- **Document ID:** `wikipedia:5910567` | **Source Locator:** [https://en.wikipedia.org/?curid=5910567](https://en.wikipedia.org/?curid=5910567)
- **Document Title:** *Henry Scriven*
- **Question:** In the context of 'Henry Scriven', what factual principle is described?
- **Ground Truth Answer:** Henry William Scriven (born 30 August 1951) is an English Anglican bishop who has served in Europe and in America.
- **Evidence Span (`15:129`):** > "Henry William Scriven (born 30 August 1951) is an English Anglican bishop who has served in Europe and in America."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 36: `qa-test-0036` (WIKIPEDIA)
- **Document ID:** `wikipedia:61766068` | **Source Locator:** [https://en.wikipedia.org/?curid=61766068](https://en.wikipedia.org/?curid=61766068)
- **Document Title:** *Pinch Gut Creek (Brown Creek tributary)*
- **Question:** In the context of 'Pinch Gut Creek (Brown Creek tributary)', what factual principle is described?
- **Ground Truth Answer:** Pinch Gut Creek is a tributary of Brown Creek in Anson County, North Carolina that rises southeast of Polkton, North Carolina and then flows south to meet Brown Creek east of Polkton.  The watershed is about 70% forested, 23% agricultural and the rest is of other land uses.
- **Evidence Span (`41:315`):** > "Pinch Gut Creek is a tributary of Brown Creek in Anson County, North Carolina that rises southeast of Polkton, North Carolina and then flows south to meet Brown Creek east of Polkton.  The watershed is about 70% forested, 23% agricultural and the rest is of other land uses."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 37: `qa-test-0037` (WIKIPEDIA)
- **Document ID:** `wikipedia:65276893` | **Source Locator:** [https://en.wikipedia.org/?curid=65276893](https://en.wikipedia.org/?curid=65276893)
- **Document Title:** *Diogo Kopke*
- **Question:** In the context of 'Diogo Kopke', what factual principle is described?
- **Ground Truth Answer:** Diogo Kopke (1808–1844) was a Portuguese editor and publisher, mathematician, journalist and soldier of German descent. He was the first person to publish the only existing contemporary report of Vasco da Gama’s first trip to India.
- **Evidence Span (`13:245`):** > "Diogo Kopke (1808–1844) was a Portuguese editor and publisher, mathematician, journalist and soldier of German descent. He was the first person to publish the only existing contemporary report of Vasco da Gama’s first trip to India."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 38: `qa-test-0038` (WIKIPEDIA)
- **Document ID:** `wikipedia:73382991` | **Source Locator:** [https://en.wikipedia.org/?curid=73382991](https://en.wikipedia.org/?curid=73382991)
- **Document Title:** *Lynch/Oz*
- **Question:** In the context of 'Lynch/Oz', what factual principle is described?
- **Ground Truth Answer:** Lynch/Oz is a 2022 American documentary film directed and written by Alexandre O. Philippe that explores the influence of the film The Wizard of Oz (1939) on the American filmmaker David Lynch.
- **Evidence Span (`10:203`):** > "Lynch/Oz is a 2022 American documentary film directed and written by Alexandre O. Philippe that explores the influence of the film The Wizard of Oz (1939) on the American filmmaker David Lynch."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 39: `qa-test-0039` (WIKIPEDIA)
- **Document ID:** `wikipedia:7818786` | **Source Locator:** [https://en.wikipedia.org/?curid=7818786](https://en.wikipedia.org/?curid=7818786)
- **Document Title:** *Pratapgarh, Tripura*
- **Question:** In the context of 'Pratapgarh, Tripura', what factual principle is described?
- **Ground Truth Answer:** Pratapgarh is a census town in West Tripura district in the Indian state of Tripura.
- **Evidence Span (`21:105`):** > "Pratapgarh is a census town in West Tripura district in the Indian state of Tripura."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 40: `qa-test-0040` (WIKIPEDIA)
- **Document ID:** `wikipedia:81268754` | **Source Locator:** [https://en.wikipedia.org/?curid=81268754](https://en.wikipedia.org/?curid=81268754)
- **Document Title:** *Damhat Sisamci*
- **Question:** In the context of 'Damhat Sisamci', what factual principle is described?
- **Ground Truth Answer:** Damhat Sisamci (born 1993) is a German politician serving as a member of the Landtag of Saarland since 2022. He is the chairman of Jusos in Neunkirchen.
- **Evidence Span (`16:168`):** > "Damhat Sisamci (born 1993) is a German politician serving as a member of the Landtag of Saarland since 2022. He is the chairman of Jusos in Neunkirchen."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 41: `qa-test-0041` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0001` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0001](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0001)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #1)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #1)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 200.
- **Evidence Span (`69:700`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 200."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 42: `qa-test-0042` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0005` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0005](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0005)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #5)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #5)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 204.
- **Evidence Span (`69:700`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 204."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 43: `qa-test-0043` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0009` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0009](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0009)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #9)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #9)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 208.
- **Evidence Span (`69:700`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 208."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 44: `qa-test-0044` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0013` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0013](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0013)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #13)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #13)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 212.
- **Evidence Span (`70:701`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 212."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 45: `qa-test-0045` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0017` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0017](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0017)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #17)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #17)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 216.
- **Evidence Span (`70:701`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 216."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 46: `qa-test-0046` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0021` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0021](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0021)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #21)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #21)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 220.
- **Evidence Span (`70:701`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 220."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 47: `qa-test-0047` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0025` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0025](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0025)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #25)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #25)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 224.
- **Evidence Span (`70:701`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 224."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 48: `qa-test-0048` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0029` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0029](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0029)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #29)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #29)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 228.
- **Evidence Span (`70:701`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 228."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 49: `qa-test-0049` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0033` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0033](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0033)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #33)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #33)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 232.
- **Evidence Span (`70:701`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 232."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`

### Item 50: `qa-test-0050` (GOVINFO)
- **Document ID:** `govinfo:GOVINFO-119-PL-42-0037` | **Source Locator:** [https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0037](https://www.govinfo.gov/app/details/GOVINFO-119-PL-42-0037)
- **Document Title:** *National Quantum and Intelligent Systems Initiative Act (Record #37)*
- **Question:** Under the legal or regulatory mandate of 'National Quantum and Intelligent Systems Initiative Act (Record #37)', what provision is stated?
- **Ground Truth Answer:** An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 236.
- **Evidence Span (`70:701`):** > "An Act to provide for a coordinated federal research programme to accelerate intelligent software systems, vector database acceleration, and quantum information science. Be it enacted by the Senate and House of Representatives of the United States of America in Congress assembled. Section 101: The Director of the National Science Foundation shall establish a competitive grant programme for academic consortia evaluating data-intensive intelligent systems, latency characterisation under heavy load, and robust artificial intelligence architectures. Enacted and recorded in official statutory register under volume 142, page 236."
- **Verification Checklist:**
  - [ ] Factually supported by evidence text
  - [ ] Title leakage identified / flagged
  - [ ] Evidence offsets verified against raw text
  - **Signoff:** `[ ] Approved  [ ] Needs Paraphrase  [ ] Discard`
