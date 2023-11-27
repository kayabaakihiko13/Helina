import unittest
from Hela.common import antidifferential as antidifferential
from math import sqrt


class TestAntriDifferential(unittest.TestCase):
    def test_antidifferential_RiemannSum(self):
        def formula(x):
            return x**2

        result = antidifferential.general_antiderivative(formula, 1, 4, 100)
        self.assertAlmostEqual(result, 20.775, places=1)

    def test_antidifferential_trapezoidal(self):
        def formula(x):
            return x**2

        result = antidifferential.general_antiderivative(
            formula, 0, 3, method="trapezoidal"
        )
        self.assertAlmostEqual(result, 9.28, places=1)

    def testing_powerRule_antidifferential(self):
        result = antidifferential.PowerRule_antiderivative(n=3, x=2)
        self.assertEqual(result, 4.0)

    def testing_subtitution(self):
        def g_formula(x):
            return sqrt(x**4 + 11)

        def f_formula(x):
            return x**3 * g_formula(x)

        result = antidifferential.Usubstitution(f_formula, g_formula, 1, 2)
        self.assertAlmostEqual(result, 3182.5855488, places=2)
