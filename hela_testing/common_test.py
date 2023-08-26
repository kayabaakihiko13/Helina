import unittest
import math
from Hela.common.common import (
    Sigmoid,
    Gaussian,
    ContinousDistribution,
    BetaDistribution,
)


class SigmoidTest(unittest.TestCase):
    def test_sigmoid_function(self):
        value = [-1.0, 1.0, 2.0]
        expected_result = list(1 / (1 + math.exp(-vector)) for vector in value)
        actual_result = Sigmoid(value).calculate_sigmoid()
        self.assertEqual(actual_result, expected_result)


class GaussianTest(unittest.TestCase):
    def test_calculate_gaussian(self):
        gaussian_calc = Gaussian(x=0.0)
        result = gaussian_calc.calculate_gaussian()
        self.assertAlmostEqual(result, 0.3989422804014337, places=6)

    def test_custom_parameter(self):
        gaussian_calc = Gaussian(x=2.0, mu=1.0, sigma=0.5)
        result = gaussian_calc.calculate_gaussian()
        self.assertAlmostEqual(result, 0.3441465248732082, places=6)

    def test_repr(self):
        gaussian_calc = Gaussian(x=2.0, mu=1.0, sigma=0.5)
        representation = repr(gaussian_calc)
        self.assertEqual(representation, "Gaussian(x=2.0, mu=1.0, sigma=0.5)")


class TestContinuousDistribution(unittest.TestCase):
    def test_continous_uniform_pdf(self):
        pdf = ContinousDistribution.continous_uniform_pdf(3.0, 2.0, 5.0)
        self.assertAlmostEqual(pdf, 0.333333, places=6)

    def test_continous_uniform_cdf(self):
        cdf = ContinousDistribution.continous_uniform_pdf(3.0, 2.0, 5.0)
        self.assertAlmostEqual(cdf, 0.333333, places=6)

    def test_generate_random(self):
        sample = ContinousDistribution.generate_random_sample(2.0, 5.0)
        self.assertTrue(2.0 <= sample <= 5.0)


class TestBetaDistribution(unittest.TestCase):
    def test_beta_pdf(self):
        pdf = BetaDistribution.beta_pdf(0.3, alpha=2, beta=5)
        self.assertAlmostEqual(pdf, 2.1608999999999994, places=6)

    def test_beta_cdf(self):
        cdf = BetaDistribution.beta_cdf(0.3, alpha=2, beta=5)
        self.assertAlmostEqual(cdf, 0.5787416212509751, places=6)
