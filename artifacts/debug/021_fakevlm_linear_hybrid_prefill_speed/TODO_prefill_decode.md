# TODO: FakeVLM Prefill-Decode Hybrid Speed

Prefill-decode support is intentionally not implemented in this debug run.

Needed work:

- Add a decode timing path that separates the initial full prompt forward from token-by-token cached decode.
- Extend policy format with `selected_decode_backend` and transition/conversion costs where needed.
- Include `marlin_nvfp4` and `dense_nvfp4_prefill_marlin_decode` candidates for decode-sensitive routing.
- Validate that FakeVLM generation outputs and cache semantics remain compatible with any per-linear backend transition.
