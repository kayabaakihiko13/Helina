import unittest

from math_testing import GeometricMean

if __name__ == "__main__":
    # Create instances of your test classes
    math_test_suite = unittest.TestLoader().loadTestsFromTestCase()

    # Combine the test suites
    all_tests = unittest.TestSuite([math_test_suite])

    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(all_tests)
