import unittest

import torch

from daso.modeling.residual_quality import fit_residual_quality


class ResidualQualityTests(unittest.TestCase):
    def test_signed_correction_and_unseen_combination(self):
        x = torch.tensor([[1., 0.], [0., 1.]])
        fit = fit_residual_quality(x, torch.tensor([-.02, .03]), torch.zeros(2), regularization=1e-8)
        result = fit.predict(torch.tensor([[0., 0.], [1., 1.]]), torch.zeros(2))
        self.assertEqual(float(result[0]), 0.)
        self.assertAlmostEqual(float(result[1]), .01, places=7)
        self.assertLess(float(fit.coefficients[0]), 0.)

    def test_policy_and_action_sums_agree(self):
        actions = torch.tensor([[[0., 0.], [1., 0.]], [[0., 0.], [0., 1.]]], dtype=torch.float64)
        fit = fit_residual_quality(torch.eye(2), torch.tensor([-.02, .03]), torch.zeros(2), regularization=.01)
        selected = actions[torch.arange(2), torch.tensor([1, 1])]
        direct = fit.predict(selected.sum(0), torch.tensor(.04, dtype=torch.float64))
        additive = (selected @ fit.coefficients).sum() + .04
        self.assertAlmostEqual(float(direct), float(additive), places=9)

    def test_rejects_misaligned_data(self):
        with self.assertRaises(ValueError):
            fit_residual_quality(torch.ones(2, 3), torch.ones(3), torch.ones(2), regularization=.1)


if __name__ == '__main__':
    unittest.main()
