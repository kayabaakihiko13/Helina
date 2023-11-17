import unittest
import Hela.mathfunc as mathfunc


class TestNthRootFunction(unittest.TestCase):
    def test_real_number(self):
        result = mathfunc.nthroot(8, 3)
        self.assertEqual(result, 2.0)

    def test_complex_number(self):
        result = mathfunc.nthroot(27, 3)
        self.assertAlmostEqual(result, 3.0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            mathfunc.nthroot("invalid", 2)
