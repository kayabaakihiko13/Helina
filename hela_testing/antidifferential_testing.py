import unittest
from Hela.common.antidifferential import AntiDifferential


class Testing_AntiDifferential(unittest.TestCase):
    def setUp(self):
        self.intergral = AntiDifferential()

    def test_antidifferential_RiemannSum(self):
        formula = lambda x: x**2
        result = self.intergral.general_antiderivative(formula, 1, 4, 100)
        self.assertEqual(result, 21.0)

    def test_antidifferential_trapezoidal(self):
        formula = lambda x: x**2
        result = self.intergral.general_antiderivative(
            formula, 0, 3, method="trapezoidal"
        )
        self.assertEqual(result, 9.0)

    def testing_powerRule_antidifferential(self):
        result = self.intergral.PowerRule_antiderivative(n=3, x=2)
        self.assertEqual(result, 4.0)
