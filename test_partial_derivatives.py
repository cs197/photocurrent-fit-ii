
import unittest

from partial_derivatives import make_d_d_eta


def rock_paper_scissors(eta, v0):
    return 0.5 * eta * eta + 0.5 * eta * eta * v0 * v0


class TestPartialDerivatives(unittest.TestCase):

    # Test the partial derivative code
    def test_take_d_d_eta(self):

        d_d_eta = make_d_d_eta(rock_paper_scissors)

        # We are expecting this to be eta + eta * v0 * v0, approximately.

        eta = 0.5
        v0 = 2.0

        result = d_d_eta(eta, v0)

        expected_result = eta + eta * v0 * v0

        self.assertAlmostEqual(expected_result, result, 2)
