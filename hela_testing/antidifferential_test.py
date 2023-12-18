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
        result = antidifferential.PowerRule_antiderivative(3, 2)
        self.assertEqual(result, 4.0)

    def testing_subtitution(self):
        def g_formula(x):
            return sqrt(x**4 + 11)

        def f_formula(x):
            return x**3 * g_formula(x)

        result = antidifferential.Usubstitution(f_formula, g_formula, 1, 2)
        self.assertAlmostEqual(result, 3182.5855488, places=2)


class testingSymmetry(unittest.TestCase):
    def General_Symmetry(self):
        def f(x):
            return x**2

        points = (0, 5)
        expected_result = 41.666666666666664
        result = antidifferential.Symmetry_integral(f, points)

        # Perform the assertion
        self.assertAlmostEqual(result, expected_result, places=10)

    def Symmetry_negative_test(self):
        def f(x):
            return x**3

        points = (-1, 5)

        expected_result = 12.7

        result = antidifferential.Symmetry_integral(f, points)
        self.assertAlmostEqual(result, expected_result, places=10)


class TestPartialIntegral(unittest.TestCase):
    def test_partial_integral(self):
        def f(x):
            return x**2

        def g(x):
            return x**3

        a = 1
        b = 3

        expected_result = (
            f(b) * g(b) - f(a) * g(a) - antidifferential.Usubstitution(f, g, a, b)
        )
        result = antidifferential.partialIntegral(f, g, a, b)

        self.assertAlmostEqual(
            result,
            expected_result,
            places=7,
            msg="Partial integral calculation is incorrect",
        )
