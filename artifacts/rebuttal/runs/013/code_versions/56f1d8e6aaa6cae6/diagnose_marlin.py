"""Check unchanged Marlin against its quantized reference and repeated outputs."""
import torch
from common import RUN, write_json
from int4_marlin import pack, forward
from standalone_marlin_int4.marlin_int4.linear import _make_workspace

torch.manual_seed(146013)
weight = torch.randn(6144, 4096, device='cuda', dtype=torch.bfloat16) * .02
packed, scales, reference = pack(weight)
workspace = _make_workspace('cuda', 6144)
results = []
for m in [8, 16384]:
    x = torch.randn(m, 4096, device='cuda', dtype=torch.bfloat16)
    expected = x @ reference.T
    baseline = forward(x, packed, scales, workspace, 6144, 4096).clone()
    errors, different = [], []
    for index in range(12):
        actual = forward(x, packed, scales, workspace, 6144, 4096)
        errors.append(float(((actual.float() - expected.float()).square().mean() / expected.float().square().mean()).sqrt()))
        different.append(int(torch.count_nonzero(actual != baseline)))
    assert max(errors) < .02, errors
    results.append({'m': m, 'n': 6144, 'k': 4096, 'relative_rmse_to_quantized_reference': errors,
        'different_bf16_values_from_first': different, 'elements': actual.numel()})
write_json(RUN / 'results/marlin_repeat_probe.json', {'passed_reference_check': True,
    'unchanged_kernel': True, 'results': results, 'scope': 'Two actual Linear shapes; no inference-policy or kernel changes.'})
print(results, flush=True)
