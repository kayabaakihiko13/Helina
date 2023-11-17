import unittest
import Hela.mathfunc as mathfunc


class TestPiFunction(unittest.TestCase):
    def test_tanpi_real(self):
        result = mathfunc.tanpi(0.25)
        expected = 1.0
        self.assertAlmostEqual(result, expected)

    def test_tanpi_complex(self):
        result = mathfunc.tanpi(0.25)
        expected = 1.0
        self.assertAlmostEqual(result, expected, places=4)

    def test_tanpi_overflow(self):
        with self.assertRaises(OverflowError):
            mathfunc.tanpi(10**1000)

    def test_cotpi_real(self):
        result = mathfunc.cotpi(0.25)
        expected = 1.000
        self.assertAlmostEqual(result, expected, places=4)

    def test_cotpi_overflow(self):
        with self.assertRaises(OverflowError):
            mathfunc.cotpi(10**1000)
