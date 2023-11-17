import unittest
import Hela.mathfunc as mathfunc


class TestErfcFunction(unittest.TestCase):
    def test_erf_taylor(self):
        self.assertAlmostEqual(mathfunc._erf_taylor(0.5), 0.5204998778130465, places=7)

    def test_erfc_mid(self):
        self.assertAlmostEqual(mathfunc._erfc_mid(0.5), 0.4795001221869535, places=7)

    def test_erfc_asymp(self):
        self.assertAlmostEqual(mathfunc._erfc_asymp(0.5), 1114111558528938.4, places=7)

    def test_erf(self):
        self.assertAlmostEqual(mathfunc.erf(0.5), 0.5204998778130465, places=7)

    def test_erfc(self):
        self.assertAlmostEqual(mathfunc.erfc(0.5), 0.4795001221869535, places=7)
