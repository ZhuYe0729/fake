## 2026-07-20 - vLLM runtime paper draft
- Development purpose: prepare a focused LaTeX draft for the runtime integration and overlap ablation described in `wu.tex`.
- Planned output: `tmp.tex`, containing only the replacement method paragraph, runtime-ablation table, and accompanying experiment paragraph with result placeholders.

## 2026-07-20 - Runtime section draft
- Added `tmp.tex` with a paper-ready English vLLM runtime subsection and a compact overlap/memory-lifecycle ablation table.
- The text uses only result placeholders, describes the implemented per-Linear copy-stream/CUDA-event protocol, and scopes the runtime claim to offline sequential TP=1 execution.

## 2026-07-20 - Condensed overlap-focused version
- Replaced the long subsection and table with one concise method paragraph focused on the asynchronous per-Linear overlap design.
- Kept only a comment identifying the intended synchronous-switch comparison for the experimental ablation.
