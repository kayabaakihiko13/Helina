import unittest
from Hela.euler import Euler


class EulerTotientTest(unittest.TestCase):
    def test_totient_of_positive_number(self):
        result = Euler().totient(10)
        expected = [-1, 0, 1, 2, 2, 4, 2, 6, 4, 6, 9]
        self.assertEqual(result, expected)

    def test_totient_of_zero(self):
        result = Euler().totient(0)
        expected = [-1]
        self.assertEqual(result, expected)
