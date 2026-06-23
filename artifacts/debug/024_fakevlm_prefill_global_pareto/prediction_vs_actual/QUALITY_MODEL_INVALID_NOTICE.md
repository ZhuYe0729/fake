# Quality Model Invalid Notice

The current v1 FakeVLM NLL-based quality artifacts are invalid and must not be used as quality-model evidence.

## Root Cause

- The LLaVA tokenizer uses left padding and expands `<image>` into a long image-token sequence.
- The old loss dataset treated `prompt_len:full_len` as an absolute right-padded answer span.
- The resulting labels targeted image tokens rather than assistant answer tokens.
- The separately encoded prompt also ended with EOS and was not a strict prefix of the full sequence.

## Evidence

- Old dense NLL: approximately `13.809`; 39 of 61 stratified deltas were negative.
- Inspecting old active labels showed repeated token `32000` (`<image>`), not answer text.
- With active-token prefix alignment, 100/100 inspected samples decoded to the assistant answer and contained no image tokens.
- A 32-sample GPU smoke changed dense NLL to `0.500791`; the extreme P30 policy increased it to `1.112333` (`+0.611542`), restoring the expected degradation direction.

## Impact

- Invalid: old stratified NLL, fitted quality coefficients, quality costs, NLL-selected Pareto policies, and quality prediction plots derived from them.
- Still valid independently: single-linear latency model comparisons and measured speed data.

The corrected loss definition is `assistant_answer_token_nll_v2_active_prefix_aligned`.
