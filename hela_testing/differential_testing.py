import unittest

from Hela.common.differential import Differential


class TestDifferential(unittest.TestCase):
    def setUp(self):
        self.diff = Differential()

    def test_derivative(self):
        def f(x):
            return x**2

        result = self.diff.derivative(f, 2)
        self.assertAlmostEqual(result, 4.0000)

    def test_power_derivative(self):
        result = self.diff.power_derivative(3, 2)
        self.assertEqual(result, 12.0)

    def test_product_derivative(self):
        def f(x):
            return x**2

        def g(x):
            return x**3

        result = self.diff.product_derivate(f, g, 2)
        self.assertEqual(result, 80.0)

    def test_quotient_derivate(self):
        f = lambda x : x ** 2
        h = lambda x : x

        result = self.diff.quotient_derivate(f,h,2)
        self.assertEqual(result, -1.0)
 
    def test_composite_derivative(self):
        def g(x):
            return x**3

        def k(x):
            return x**2 + 1

        result = self.diff.composite_derivative(g, k, 2)
        self.assertEqual(result, 300.0)
