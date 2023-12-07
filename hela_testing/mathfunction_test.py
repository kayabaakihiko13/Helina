import unittest
import Hela.mathfunc as math_function


def square_function(x):
    return x**2


def cubic_function(x):
    return x**3


def square_complex(x):
    return x**2 + 1j


class TestMathFunctionReal(unittest.TestCase):
    def test_real_input(self):
        math_function_real = math_function._mathfunction_real(
            square_function, square_complex
        )
        result_real = math_function_real(3.0)
        self.assertEqual(result_real, 9.0)

    def test_invalid_input(self):
        math_function_invalid = math_function._mathfunction_real(
            square_function, square_complex
        )
        with self.assertRaises(ValueError):
            result_invalid = math_function_invalid("invalid_input")


class TestPolyvalFunction(unittest.TestCase):
    def test_polyval_real(self):
        coefficients = [2, 3, 1]
        self.assertAlmostEqual(math_function._polyval(coefficients, 2), 15)

    def test_polyval_complex(self):
        complex_coefficients = [1 + 2j, 2 - 3j, 1 - 1j]
        result = math_function._polyval(complex_coefficients, 1 + 1j)
        expected_res = 2 + 0j
        self.assertAlmostEqual(result.real, expected_res.real)
        self.assertAlmostEqual(result.imag, expected_res.imag)


class TestZetaAndE1Function(unittest.TestCase):
    def test_e1(self):
        expected_result = 0.048900510708060896
        self.assertAlmostEqual(math_function.e1(2.0), expected_result, places=4)

    def test_zeta(self):
        self.assertAlmostEqual(math_function.zeta(2.0), 1.6449330668482265, places=4)

    def test_zeta_exception(self):
        with self.assertRaises(ValueError):
            math_function.zeta(1)


class TestGCD(unittest.TestCase):
    def test_positive_number(self):
        result = math_function.gcd(121, 11)
        self.assertEqual(result, 11)

    def test_assert(self):
        with self.assertRaises(TypeError):
            math_function.gcd(10, "20")


class TestModDivision(unittest.TestCase):
    def test_valid_input(self):
        result = math_function.mod_division(4, 11, 6)
        self.assertEqual(result, 1)
