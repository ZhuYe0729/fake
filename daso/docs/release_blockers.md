# Public release blockers

The standalone source and archived results are usable internally, but the
following items must be closed before advertising complete public GPU
reproduction:

1. Review and publish the three final vLLM working-tree changes listed in
   `external_backends.md`, then replace the base revision with a clean release
   commit.
2. Select a project license after confirming ownership and compatibility with
   all reused third-party material.  `LICENSE_PENDING.md` intentionally grants
   no rights.
3. Supply the publisher style, bibliography, `motivation.pdf`, and
   `overview3.pdf` if building the complete paper PDF is a release goal.
4. Optionally rerun Llama-2 generation BF16, dense-NVFP4, and sparse-BF16 to
   restore per-sample evidence.  Their surviving aggregate measurements are
   preserved and explicitly identified in the provenance report.

The vLLM and CUTLASS repositories remain external by design; model weights and
datasets are not redistributed.
