# FakeVLM

Supports compression of language-model Linear layers while excluding the vision
tower, multimodal projector, and LM head. FakeClue data and the upstream
FakeVLM repository are external dependencies.

The `fakevlm` selector in `src/daso/compression/selectors.py` preserves the
compression boundary. `model.py`, `dataset.py`, and `evaluate.py` retain the
dense model adapter, multimodal collation, and deterministic FakeClue scoring
without vendoring the upstream repository or checkpoint.
