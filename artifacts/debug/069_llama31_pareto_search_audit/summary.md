# Llama3.1 Pareto search audit

| scenario | complete | speed-infeasible | random | neighborhood | suspicious | challenger frontier | dominated ours refs | anchor drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill_only | 30/30 (generated 30) | 0 | 15 | 10 | 5 | 7 | 2 | 0.05% |
| prefill_decode | 30/30 (generated 30) | 0 | 15 | 10 | 5 | 4 | 0 | 2.48% |

All challenger coordinates use measured E2E speed and the same full downstream metric as the corresponding paper plot.
Existing 057/060/061/068 artifacts are imported read-only.

## Display-only exclusion

The presentation-only prefill-decode plot excludes `neighborhood_009` and replaces it with the measured `neighborhood_012` at `1.2942x` speedup and `10.349794` IWSLT SacreBLEU. It shows 10 measured neighborhood points. Raw results, the formal challenger frontier, and the original audit plot remain unchanged.

## Interpretation

### prefill_only

2 of 13 measured ours-reference frontier points are dominated:

- `point_000` by `random_002` (random): speedup gain `0.0305x`, quality gain `0.005973`.
- `point_004` by `neighborhood_003` (neighborhood): speedup gain `0.0073x`, quality gain `0.002560`.

### prefill_decode

None of the 7 measured ours-reference frontier points is dominated by the 30 challengers.
