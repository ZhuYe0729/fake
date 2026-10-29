import unittest

import torch

from daso.modeling.quality_constraint import forecast_quality_guard


class QualityConstraintTests(unittest.TestCase):
    def test_paired_correlations_cancel(self):
        anchor = torch.tensor([-.03, -.01, .01, .03], dtype=torch.float64)
        result = forecast_quality_guard(anchor, -anchor[None], torch.tensor([[0], [1]]), validation_windows=128)
        self.assertGreater(result.standard_error[0], 0)
        self.assertLess(abs(float(result.standard_error[1])), 1e-10)
        self.assertAlmostEqual(float(result.guarded[1]), .005)

    def test_matches_direct_window_calculation_and_margin(self):
        anchor = torch.tensor([-.01, -.03, -.02, .01], dtype=torch.float64)
        groups = torch.tensor([[.01, .02, 0., .01], [-.02, .03, .01, -.01]], dtype=torch.float64)
        mask = torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.float64)
        result = forecast_quality_guard(anchor, groups, mask, validation_windows=4, residual_margin=.008)
        direct = anchor[None] + mask @ groups
        self.assertTrue(torch.allclose(result.mean, direct.mean(1)))
        self.assertTrue(torch.allclose(result.standard_error, direct.std(1) / 2))
        self.assertTrue(torch.allclose(result.guarded, direct.mean(1) + direct.std(1) + .013))

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "window"):
            forecast_quality_guard(torch.ones(2), torch.ones(1, 3), torch.ones(1), validation_windows=128)
        with self.assertRaisesRegex(ValueError, "binary"):
            forecast_quality_guard(torch.ones(3), torch.ones(1, 3), torch.tensor([.5]), validation_windows=128)


if __name__ == '__main__':
    unittest.main()
