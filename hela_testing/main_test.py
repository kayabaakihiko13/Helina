import unittest

from hela_testing.math_testing import (
    TestGeometricMean,
    TestPowerIteration,
    TestHarmonic,
    TestFastFourierTransforms,
)

from hela_testing.common_test import (
    SigmoidTest,
    GaussianTest,
    TestContinuousDistribution,
    TestBetaDistribution,
)

if __name__ == "__main__":
    # Create instances of your test classes
    power_iteration_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestPowerIteration
    )
    geometric_mean_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGeometricMean
    )
    harmonic_mean_test = unittest.TestLoader().loadTestsFromTestCase(TestHarmonic)

    fft_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFastFourierTransforms
    )

    sigmoid_test_suite = unittest.TestLoader().loadTestsFromTestCase(SigmoidTest)
    gaussian_test_suite = unittest.TestLoader().loadTestsFromTestCase(GaussianTest)
    test_continous_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestContinuousDistribution
    )
    test_beta_suite = unittest.TestLoader().loadTestsFromTestCase(TestBetaDistribution)

    # Combine the test suites
    all_tests = unittest.TestSuite(
        [
            power_iteration_suite,
            geometric_mean_suite,
            harmonic_mean_test,
            fft_test_suite,
            sigmoid_test_suite,
            gaussian_test_suite,
            test_continous_suite,
            test_beta_suite,
        ]
    )

    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(all_tests)
