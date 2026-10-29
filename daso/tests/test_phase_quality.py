import unittest

import torch

from daso.modeling.phase_quality import combine_phase_quality


class PhaseQualityTests(unittest.TestCase):
    def test_default_preserves_every_joint_action(self):
        pre = torch.tensor([[0., .02]], dtype=torch.float64)
        dec = torch.tensor([[0., .005, .01]], dtype=torch.float64)
        result = combine_phase_quality(pre, dec)
        self.assertEqual(tuple(result.shape), (1, 2, 3))
        torch.testing.assert_close(result, pre[:, :, None] + dec[:, None, :])
        self.assertEqual(float(result[0, 0, 0]), 0.)

    def test_prefill_weight_changes_preference_without_removing_actions(self):
        pre = torch.tensor([[0., .02]], dtype=torch.float64)
        dec = torch.tensor([[0., .005]], dtype=torch.float64)
        latency = torch.tensor([[6., 5., 3., 2.]])
        choices = []
        for weight in [1., 2.]:
            q = combine_phase_quality(pre, dec, prefill_weight=weight).reshape(1, -1)
            feasible = q <= .025 + 1e-10
            choices.append(int(torch.where(feasible, latency, float('inf')).argmin()))
        self.assertEqual(choices, [3, 1])

    def test_rejects_invalid_weight(self):
        with self.assertRaises(ValueError):
            combine_phase_quality(torch.zeros(1, 2), torch.zeros(1, 2), prefill_weight=0.)


if __name__ == '__main__':
    unittest.main()
