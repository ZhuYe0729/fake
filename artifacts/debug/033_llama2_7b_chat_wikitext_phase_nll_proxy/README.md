# Llama2 WikiText phase-NLL proxy debug

This is an isolated quality-model debug experiment. It does not modify exported Pareto results and does not solve a Pareto frontier.

- 72 controlled phase policies, split into 54 train and 18 holdout policies.
- WikiText-2 blocks use 2048 prompt tokens and 80 teacher-forced decode tokens.
- PMPD is reserved for six external transfer checks after fitting.

The primary expected outcome is a positive and stable prefill-decode holdout ranking. If it is not obtained, this directory records the failure rather than promoting the proxy.
