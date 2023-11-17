import unittest
import Hela.mathfunc as mathfunc


class TestGammaFunction(unittest.TestCase):
    def test_positive_gamma_function(self):
        result = mathfunc._gamma_real(5)
        self.assertAlmostEqual(result, 24.0, places=4)

    def test_gamma_fraction(self):
        result = mathfunc._gamma_real(0.5)
        self.assertAlmostEqual(result, 1.77245385091, places=4)

    def test_gamma_zero(self):
        with self.assertRaises(ZeroDivisionError):
            mathfunc._gamma_real(0)

    def test_gamma_negative_integers(self):
        with self.assertRaises(ZeroDivisionError):
            mathfunc._gamma_real(-2)

    def test_gamma_large_value(self):
        result = mathfunc._gamma_real(10)
        self.assertAlmostEqual(result, 362880.0, places=4)

    def test_rgamma(self):
        self.assertAlmostEqual(mathfunc._gamma_real(5), 24.0)
        self.assertAlmostEqual(mathfunc._gamma_real(0.5), 1.77245385091)
        with self.assertRaises(ZeroDivisionError):
            mathfunc._gamma_real(-2)

    def test_gamma_complex(self):
        result = mathfunc._gamma_complex(5)
        self.assertAlmostEqual(result.real, 24.0)
        self.assertAlmostEqual(result.imag, 0.0)
