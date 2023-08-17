import unittest
import numpy as np
from Hela.math import GeometricMean, PowerIteration


class TestGeometricMean(unittest.TestCase):
    def test_geometric_mean(self):
        gm = GeometricMean()
        self.assertAlmostEqual(gm([2, 4, 8]), 4.0)
        self.assertAlmostEqual(gm([1, 2, 3, 4, 5]), 2.605171084697352)
        with self.assertRaises(ValueError):
            gm([])
        with self.assertRaises(ValueError):
            gm("invalid input")
