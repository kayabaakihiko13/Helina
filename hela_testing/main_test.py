import unittest

from hela_testing.math_testing import TestGeometricMean, TestPowerIteration, TestHarmonic

if __name__ == "__main__":
    # Create instances of your test classes
    power_iteration_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestPowerIteration
    )
    geometric_mean_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGeometricMean
    )
    harmonic_mean_test = unittest.TestLoader().loadTestsFromTestCase(TestHarmonic)

    # Combine the test suites
    all_tests = unittest.TestSuite(
        [power_iteration_suite, geometric_mean_suite, harmonic_mean_test]
    )

    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(all_tests)
