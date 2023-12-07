import unittest

from Hela.common import differential


class TestDifferential(unittest.TestCase):
    def test_derivative(self):
        def f(x):
            return x**2

        result = differential.derivative(f, 2)
        self.assertAlmostEqual(result, 4.0000)

    def test_power_derivative(self):
        result = differential.power_derivative(3, 2)
        self.assertEqual(result, 12.0)

    def test_product_derivative(self):
        def f(x):
            return x**2

        def g(x):
            return x**3

        result = differential.product_derivate(f, g, 2)
        self.assertEqual(result, 80.0)

    def test_quotient_derivate(self):
        f = lambda x: x**2
        h = lambda x: x

        result = differential.quotient_derivate(f, h, 2)
        self.assertEqual(result, -1.0)

    def test_composite_derivative(self):
        def g(x):
            return x**3

        def k(x):
            return x**2 + 1

        result = differential.composite_derivative(g, k, 2)
        self.assertEqual(result, 300.0)
