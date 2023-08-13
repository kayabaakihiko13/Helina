import unittest
from math_testing import MathTest
from euler_test import EulerTotientTest

if __name__ == "__main__":
    # Create instances of your test classes
    math_test_suite = unittest.TestLoader().loadTestsFromTestCase(MathTest)
    euler_test_suite = unittest.TestLoader().loadTestsFromTestCase(EulerTotientTest)

    # Combine the test suites
    all_tests = unittest.TestSuite([math_test_suite, euler_test_suite])

    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(all_tests)
