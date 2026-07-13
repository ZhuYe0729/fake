# 088 Llama2 WikiText Pareto Validation Plan

## Objective
- Validate selected predicted frontier points before any promotion beyond debug.

## Decisions
- Select five points per scenario: dense endpoint, predicted fastest endpoint, knee, and two evenly spaced intermediates.
- First measure true 100-block WikiText phase NLL for all selected policies.
- Then export the same policies and measure vLLM E2E under each established scenario protocol.
- Keep PMPD transfer evaluation out of this pass; it follows only if the selected points preserve WikiText ordering.

## Verification
- Compare predicted versus measured NLL rank and absolute deltas.
- Compare raw predicted linear latency versus measured vLLM latency without fitting a correction table.
- Flag non-monotonic measured points rather than silently dropping them.
