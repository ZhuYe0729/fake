# Third-party components

DASO interoperates with, but does not vendor, the following projects:

- vLLM and the separately maintained phase-heterogeneous extension:
  <https://github.com/ZhuYe0729/vllm-cospaq>.
- NVIDIA CUTLASS and the separately maintained DASO wrapper:
  <https://github.com/ZhuYe0729/cutlass_wrapper>.
- Hugging Face Transformers and Datasets, under their upstream licenses.
- EleutherAI lm-evaluation-harness, pinned to 0.4.12 for paper evaluation.
- SparseGPT/OBC for Hessian-aware pruning concepts and reference behavior.
- DINOv3, MIRROR, FakeVLM, and timm for optional vision-model examples.

Users must install these projects under their respective licenses. Exact
runtime revisions used by a run are captured by the preflight manifest.
No third-party license grant is implied by DASO's pending project license.
