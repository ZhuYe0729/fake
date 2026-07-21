# Real vLLM teacher-forced prefill-decoding NLL

This experiment replaces the historical HF/proxy prefill-decoding NLL with
real vLLM execution.  A logits processor records each target token's
pre-processor log probability and then forces that target token as the next
decode input.  Therefore every scored continuation token executes the actual
decode route while retaining the reference continuation (teacher forcing).

The initial probe must verify that this mechanism works with vLLM V1 before
the two-model policy inventory is evaluated.  All results are new files in
this directory; historical NLL, speed, and generation-task artifacts are not
modified.
