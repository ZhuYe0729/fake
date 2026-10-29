import unittest
import torch

from daso.modeling.conditional_quality import fit_conditional_quality


class ConditionalQualityTests(unittest.TestCase):
    def test_anchor_signed_deltas_and_unseen_combination(self):
        reference = torch.tensor([2., 3., 4.], dtype=torch.float64)
        anchor = reference - .02
        singles = torch.stack([anchor + .01, anchor - .005, anchor + .03])
        fit = fit_conditional_quality(reference, anchor, singles)
        decisions = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
        self.assertTrue(torch.allclose(fit.predict(decisions), torch.tensor([-.02, -.01, -.025, .015], dtype=torch.float64)))
        self.assertLess(fit.group_deltas[1], 0)


    def test_pair_correction_and_batch_predictions(self):
        reference = torch.tensor([2., 3., 4., 5.], dtype=torch.float64)
        anchor = reference - .01
        singles = torch.stack([anchor + .02, anchor + .03, anchor + .01])
        pair = (anchor + .09)[None]
        fit = fit_conditional_quality(reference, anchor, singles, pairs=((0, 1),), pair_nll=pair)
        # The extra .04 is an interaction, not another single-group penalty.
        actual = fit.predict(torch.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1]]))
        self.assertTrue(torch.allclose(actual, torch.tensor([.02, .08, .09], dtype=torch.float64)))


    def test_aligned_window_shapes_are_required(self):
        with self.assertRaisesRegex(ValueError, "window"):
            fit_conditional_quality(torch.ones(3), torch.ones(4), torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, "pair"):
            fit_conditional_quality(torch.ones(3), torch.ones(3), torch.ones(2, 3), pairs=((0, 1),))


if __name__ == '__main__':
    unittest.main()
