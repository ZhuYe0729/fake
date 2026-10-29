# AK6a

**Paper Summary:**

This paper proposes DASO, a framework for automatically selecting compression strategies at the module and inference-phase levels. It combines hardware-aware latency prediction, quality estimation, and constrained optimization to generate deployment policies. The framework is implemented in vLLM and evaluated on different GPU platforms.

**Strengths:**

This paper addresses an important practical problem: choosing appropriate compression methods for different deployment requirements. Combining module-wise and phase-wise decisions is a useful direction. The integration with an actual inference runtime is a strong point, and the reported speedups suggest that the approach can be effective in practice.

**Weaknesses:**

My main concern is the reliability of the quality constraints. DASO uses an approximate quality model based on local compression perturbations, but it is not entirely clear how accurately the final selected policies satisfy the specified quality budgets. This seems particularly important when the allowed quality loss is small.

Most of the main comparisons are against uniform compression configurations. Since heterogeneous compression is central to the paper, stronger comparisons with existing mixed-precision or layer-wise allocation methods would help establish the benefit of the proposed optimization.

I would also like to see more discussion of the memory overhead introduced by using different weight representations for prefill and decoding. The evaluation is limited to two Llama models, which makes generalization to other architectures less clear.

**Justification of Recommendation:**

I lean slightly positive because the framework is practical and the implementation results are encouraging. However, the quality-constraint guarantees and the advantage over existing heterogeneous allocation methods are not fully convincing. These are important for assessing the overall contribution.

**Specific Points of Feedback for Rebuttal:**

1. How often do the selected policies actually satisfy the requested quality budgets when evaluated with measured NLL rather than predicted values? Are constraint violations observed?
2. Could the authors clarify the additional peak memory or weight-storage overhead associated with phase-wise compression, and how this affects the reported deployment gains?

**Rating:** 6: Marginally above acceptance threshold

**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct



# Exi5

**Paper Summary:**

DASO chooses a compression method for each Linear module and each inference phase (prefill vs. decode). The candidates are BF16, NVFP4, 2:4 sparse BF16, 2:4 sparse NVFP4, and W4A16 Marlin. On Llama-2-7B-Chat and Llama-3.1-8B-Instruct (RTX 5090, RTX PRO 6000), it reports up to 2.26× prefill and 1.55× end-to-end speedup over BF16, and a quality–speed frontier that sits above the uniform baselines.

**Strengths:**

- Paper is well motivated. The search space can be very large and confusing with many optimization options available. Choosing among several kernel-backed compression formats per module and per phase is a real deployment question.
- Policies are measured in a real vLLM runtime
- Table 4 and Fig. 5 test each modeling component and the calibration-set size.

**Weaknesses:**

- Production vLLM uses continuous batching with chunked prefill. One forward pass mixes prefill and decode tokens from different requests, so “prefill weights” and “decode weights” cannot be separated.
- Only one scenario is tested: B=8, S=2048, T=64. With only 64 decode steps, end-to-end time is dominated by prefill.
- The paper claims “multiple workloads” and “extensive experiments,” but the results contain two models, one batch size, one input length, and one output length.
- Missing baselines - Sensitivity-based mixed-precision allocators are dismissed as “a different design level.” HAWQ-V3, which does ILP with hardware latency and Hessian sensitivity, and HAQ are close prior work and are not cited. Their formulations are essentially the same as Eq. 11. SmoothQuant W8A8 is discussed but not used. FP8 format is missing from the candidate set.

**Justification of Recommendation:**

The problem is relevant and the system is real. However, the phase-wise premise conflicts with how modern serving batches requests, the evaluation covers a single workload and it misses several key baseline methods for a fair comparison.

**Specific Points of Feedback for Rebuttal:**

NA

**Rating:** 4: Ok but not good enough - rejection

**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct



# LrT2

**Paper Summary:**

This paper introduces DASO (Deployment-Aware Strategy Optimization), a framework that turns existing LLM compression methods (dense and sparse NVFP4, 2:4 sparse BF16, and W4A16 Marlin) into fine-grained deployment decisions. Since no single method is best for every module, inference phase, or GPU, DASO picks a method per module and per phase (prefill vs. decode) to minimize latency under a user-specified quality budget. It does this with two cost models: a latency model that combines a method-aware Roofline baseline with a learned residual correction, and a quality model that maps local output perturbations to the global ΔNLL increase using depth, module-type, and phase sensitivities. Together they reduce policy selection to a multiple-choice knapsack that dynamic programming solves in seconds. Policies run in an extended vLLM runtime that overlaps decode-weight preparation with prefill. On Llama-2-7B-Chat and Llama-3.1-8B-Instruct, using RTX 5090 with RTX PRO 6000 as a cross-platform check, DASO reaches up to 2.26× prefill and 1.55× end-to-end speedups over BF16 and gives controllable quality–speed trade-offs that beat uniform whole-model compression.

**Strengths:**

- A deployment-oriented framework that unifies multiple compression methods into a common policy space with heterogeneous module-wise and phase-wise configurations. The framework selects a deployment policy from a unified set of candidate methods, including quantization, pruning, and their combinations.
- The authors have developed a hardware-aware latency model and a model aware quality model for heterogeneous compression policies.
- It solves the policy selection by reducing the problem to a multiple-choice knapsack problem (easily solvable with dynamic programming).

**Weaknesses:**

- Tested on only Llama families which limits the DASO's generalizability validation.
- Baseline should be layer-wise or module-wise mixed-precision networks, not uniformly quantized networks.
- The accuracy evaluation of the selected latency modeling should be included in the paper. Also, the runtime overhead of latency modeling and quality modeling should be reported.

**Justification of Recommendation:**

There are many prior works that use actual hardware latency in quantization pipeline. One of many examples is "HAQ: Hardware-Aware Automated Quantization with Mixed Precision (CVPR'19)". I find the proposed deployment-oriented compression method selection framework useful, but similar frameworks have already been proposed in many academic papers (mostly, mixed-precision quantization though). To me, the novelty of such framework is less appealing considering that the evaluation is done over a single LLM family (Llama). It would be great to see how the proposed method generalizes well to other LLM families with different attention mechanisms.

Latency modeling does not include actual computing or data fetching. For the pruned network, as an example, computations involve metadata decoding latency as well which is not considered by the simple 'mnk' formulation. Perhaps this deviation is handled by the logarithmic multiplicative residual . Then, why not directly use measured latency for various compression options?

**Specific Points of Feedback for Rebuttal:**

- The accuracy evaluation on how well the proposed latency model matches with the actual latency measurement should be included in the paper. Do we need to run latency modeling on every target hardware? What will be the runtime for performing hardware-specific latency evaluation?
- The baseline should be mixed-precision networks for a fair comparison. Layer-wise mixed precision networks may provide similar speed-up against BF16 with a similar accuracy level.
- How do you model layers that cannot be converted to GEMM such as softmax or GELU? Does the latency model include these non-GEMM layers as well? Since their latencies are mostly determined by fetching the tensor, they need to be considered for more accurate delay modeling.
- Could you evaluate the framework on different LLM architectures (Qwen3 or DeepSeek models)?

**Rating:** 5: Marginally below acceptance threshold

**Confidence:** 5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature



# QboE

**Paper Summary:**

This paper introduces DASO, a deployment oriented framework that turns a set of compression methods into fine-grained deployment decisions. Specifically, DASO integrates quantization and pruning methods into a unified candidate space and automatically selects module-wise and phase-wise configurations under a user-defined quality constraint. Extensive experiments demonstrate the effectiveness of DASO.

**Strengths:**

- The proposed method is tested in actual acceleration experiments on RTX 5090 and RTX PRO 6000, demonstrating its effectiveness.
- The idea of selectively using different pruning and quantization methods based on deployment requirements is quite interesting.
- The idea of transforming the compression strategy combination problem into a multiple-choice knapsack problem and solving for the optimal combination using dynamic programming is quite novel.

**Weaknesses:**

- The formulas in Figure 2 are all garbled.
- The proposed method is only tested on 7-8B, and lacks experiments on larger scales (30B) and more diverse model structures.
- The computation time required for searching the compression strategy is missing.
- Lack of comparison with simple pruning algorithms.

**Justification of Recommendation:**

My recommendation is mainly driven by concerns about experimental completeness and reproducibility. The evaluation is limited to 7–8B models, leaving the scalability of DASO to larger models and more diverse architectures unclear.

**Specific Points of Feedback for Rebuttal:**

The rebuttal should clarify the computational overhead of the compression-strategy search, discuss the scalability of DASO beyond the evaluated 7–8B models, and explain its competitiveness relative to simpler pruning-based deployment strategies.

**Rating:** 6: Marginally above acceptance threshold

**Confidence:** 4: The reviewer is confident but not absolutely certain that the evaluation is correct



# gkP9

**Paper Summary:**

The paper introduces a framework for discovering optimal layer-wise compression policies for Large Language Models.

Recognizing that different modules contribute differently to both quality degradation and inference latency, the authors propose an optimization algorithm that consists of a latency prediction that calculates effective floating-point work and data movement for various dense, sparse, and quantized kernels. It also consider residual correction that bridges the gap between theoretical hardware priors and actual measured latency by estimating log-residuals using a mix of global ridge regression and local Euclidean interpolation. A dynamic programming solver in introduced to find a module-level compression policy to minimize end-to-end latency without exceeding a specified quality degradation budget.

The authors conclude that their approach effectively approximates the true Pareto frontier on Llama-3.1-8B-Instruct.

**Strengths:**

- The paper combines theoretical hardware priors (Roofline model tracking data movement and FLOPs) with empirical residual correction. The use of adaptive mixing between local interpolation and global regression provides a mathematically sound way to handle out-of-distribution matrix shapes.
- The paper introduces an efficient dynamic programming approach to navigate the massive combinatorial search space of module-wise compression.
- Section D tries to break the discovered Pareto frontier using random, neighborhood and leverage-guided mutations shows that 18 out of 20 DASO policies remain non-dominated by 60 blind challengers strongly supports the effectiveness of the search algorithm.

**Weaknesses:**

- The problem of searching for an optimal configuration of layer-wise operations using latency and accuracy predictors is heavily explored in the Neural Architecture Search (NAS) literature (e.g., ProxylessNAS, FBNet, BRP-NAS). The paper fundamentally proposes a NAS method applied to LLM compression but seemingly omits context, comparisons, or baseline adaptations from this adjacent field.
- Equation C.1 computes the total latency of a policy as a sum of individual module latencies. However, applying different quantization or sparsity methods (e.g., switching from Sparse NVFP4 to Dense BF16) across consecutive layers introduces hardware overheads for memory layout conversion, casting, or re-packing. The lack of discussion regarding these inter-layer transition costs threatens the real-world accuracy of the end-to-end latency predictor.
- The supplementary material lists only five distinct kernel types in the action space (Dense BF16, Sparse BF16, Dense NVFP4, Sparse NVFP4, W4A16 Marlin). This restricts the sparsity/quantization options considerably compared to state-of-the-art literature which often explores a wider variety of mixed-precision bit-widths (e.g., INT8, INT2, mixed sparsity ratios).
- The dynamic programming solver minimizes latency subject to a maximum quality degradation budget. However, in practical deployment scenarios, engineers often face the inverse problem: strict latency SLAs (TTFT requirements) where the goal is to maximize quality subject to a hard latency constraint.

**Justification of Recommendation:**

The rating is constrained by a few critical omissions. The lack of engagement with NAS literature, the limited scope of the compression action space, and the ambiguity regarding inter-layer formatting overheads prevent a higher score. If the authors can clarify the execution overheads and contextualize their work within the broader NAS framework, the paper's rating could improve.

**Specific Points of Feedback for Rebuttal:**

- Does the latency formulation in Equation C.1 account for the overhead of switching data formats/precisions between consecutive modules? If not, how significant is this overhead in the extended vLLM runtime mentioned in the text?
- The paper mentions measuring normalized output perturbation under "identical module inputs." Please clarify the methodology for this. Are you caching activations from a forward pass of the uncompressed model over a calibration dataset to use as these identical inputs?
- Can Algorithm 1 be trivially inverted to solve for maximum quality given a strict latency budget without losing the optimal substructure?
- Please comment on the relationship between your policy optimization and hardware-aware NAS.

**Rating:** 5: Marginally below acceptance threshold

**Confidence:** 4: The reviewer is confident but not absolutely certain that the evaluation is correct



#  AI Reviewer 

**Review:**

**Title:** From Compression Methods to Deployment Decisions: Quality-Constrained Fine-Grained Strategy Optimization for Efficient LLM Inference

**Synopsis of the paper**
The paper presents DASO, a framework that selects compression backends separately for model modules and inference phases under a user-specified quality constraint. It combines calibrated latency and quality models with constrained policy search and a phase-aware vLLM runtime. Experiments examine quality–speed trade-offs across Llama models, workloads, and NVIDIA GPUs.

**Summary of Review**
The paper addresses the concrete deployment problem of converting heterogeneous compression implementations into hardware- and workload-specific policies. Its distinctive contribution is joint cross-method, module-wise, and phase-wise optimization backed by an executable runtime, and the experiments demonstrate gains over same-method uniform configurations. However, the central quality-constrained claim is not verified: the headline speed maxima incur large absolute quality losses, and the selected policies’ budgets, measured quality losses, and constraint violations are not reported. The additive quality model and latency objective also omit interactions and costs that can affect which policy is feasible or optimal. Missing phase-wise hybrid baselines and closely related automated configuration work limit attribution of the gains and the stated novelty.

**Strengths**

- The deployment decision is formalized at a useful systems granularity. Equations 1 and 11 define legal backend choices per Linear module and phase and reduce selection to a tractable multiple-choice knapsack, connecting compression quality estimates to executable policies.
- The latency estimator combines a physically motivated prior with empirical shape correction. Equations 2–5 preserve method-specific compute and memory scaling, while Table 4 shows that the complete estimator lowers unseen-shape MAPE from 14.80% to 8.01%; this supports searching beyond exactly profiled shapes.
- The runtime addresses a concrete obstacle to phase-heterogeneous execution. The Runtime Deployment section introduces layer-wise asynchronous representation materialization, and Table 4 reports that only 1.92 ms of 354.94 ms cumulative transfer remains exposed for the tested policy; this makes phase-specific policies operational rather than purely simulated.
- The evaluation uses optimized kernels and full serving measurements rather than nominal compression ratios or FLOPs. Tables 2–3 cover two models, seven downstream metrics, and separately profiled GPUs, providing direct evidence about realized quality and latency.

**Weaknesses**

- The evidence does not establish DASO’s defining quality-constrained claim. The abstract’s 2.26 prefill maximum comes from the Llama-3.1 Max-speed row in Table 2, where MMLU falls from 68.40 to 31.14; its 1.55 E2E maximum comes from the Llama-2 Max-speed row, where IWSLT falls from 19.30 to 2.60. Tables 2–3 also omit , predicted and measured , and violation magnitude, so the headline results are low-quality endpoints and no selected policy is shown to satisfy the optimized budget.
- Complete-policy fitting does not provide the interaction capacity claimed for the quality model. Equations 8–9 remain additive over modules and phases, so fitting complete policies can distribute average interaction effects across marginal costs but cannot represent arbitrary interactions between compressed modules or phase choices. Table 4’s held-out MAE of 0.0413 is not interpretable without the measured  range or budget sizes, leaving the constraint model’s reliability unresolved.
- The optimized latency proxy omits a policy-dependent cost central to the runtime. Equation 10 sums only Linear-kernel latency, whereas Table 4 reports 354.94 ms of cumulative representation-transfer work and evaluates the exposed cost only for the Max-speed policy. Thus, the optimization objective is not shown to preserve policy ordering under realized E2E latency when method pairs have different switching costs.
- The novelty positioning does not fully account for automated fine-grained configuration search. AMQ already searches layer-wise mixed-precision policies using quality prediction and Pareto optimization but is not cited (Lee et al., 2025), while PMPD explicitly allocates precision by inference phase but is cited only in the sentence that it “var[ies] precision across phases or tokens” (Chen et al., 2025). DASO’s narrower novelty in cross-method backend selection and runtime deployment should therefore be distinguished more precisely.
- The comparisons cannot attribute the reported gains to module-wise optimization. Tables 2–3 include only same-method uniform policies, although dense NVFP4 provides strong prefill performance and W4A16 provides strong decoding performance; a policy applying one method uniformly during prefill and another uniformly during decoding is necessary to separate phase selection from module-level selection.
- Workload adaptability is not empirically established across workload shapes. The Experimental Setup fixes , , and  for the reported measurements, despite Equations 1 and 10 conditioning policy selection on these variables. The results therefore do not show whether policies or advantages persist across batch sizes, prompt lengths, and generation lengths.

**Suggestions for Improvement**

- A policy-composition analysis would clarify the source of DASO’s gains. Reporting backend frequencies by module type, depth, and phase across budgets and GPUs would reveal whether the optimizer discovers heterogeneous sensitivity patterns or primarily implements a global phase split.
- The method exposition would benefit from defining the configurable module set, legal phase pairs, and representation lifecycle before introducing the cost models. These concepts are currently distributed across the Framework Overview, Equations 8–10, and Runtime Deployment sections, requiring readers to reconstruct the actual action space.
- Characterizing memory footprint and serving concurrency would broaden the runtime evidence. The representation-switching pipeline stages decoding weights while prefill continues, but the paper does not quantify additional resident or transient memory, which can determine feasible batch size and throughput in deployment.

**Additional Details and Minor Issues**

- Figure 4 labels both panels “Measured E2E speedup,” although panel (a) is prefill-only and corresponds to TTFT in Table 2.
- Figure 5 reports median MAE of 0.0403 at 45 policies, while the selected 54-policy setting is reported as 0.0413 at greater cost; the distinction between these results should be explained.
- Tables 2–3 report relative speedups without absolute TTFT, TPOT, or E2E latency, limiting assessment of practical scale.
- Equation 6 omits source conditioning for summarization and translation; the likelihood definition should reflect the task input.
- Table 3 averages ROUGE-L and SacreBLEU into “Score,” which can conceal task-specific degradation.
- Several highlighted differences lack uncertainty estimates, including IWSLT scores of 10.55 versus 10.56 and TPOT speedups of 1.53 versus 1.52.

**References**

Chen, H. M., Tan, F., Kouris, A., Lee, R., Fan, H., & Venieris, S. I. (2025). Progressive mixed-precision decoding for efficient LLM inference. *International Conference on Learning Representations*. ([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5df4313ecd4875931fbdacc486cc1fcf-Abstract-Conference.html))

Lee, S., Woo, S.-t., Jin, J.-g., Lee, C., & Park, E. (2025). AMQ: Enabling AutoML for mixed-precision weight-only quantization of large language models. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 35532–35550). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.emnlp-main.1799. ([aclanthology.org](https://aclanthology.org/2025.emnlp-main.1799/))