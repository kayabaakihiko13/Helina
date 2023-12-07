import unittest

from hela_testing.common_test import (
    SigmoidTest,
    GaussianTest,
    TestBayesTheorem,
    TestReLU,
)

from hela_testing.distribution_test import (
    TestNormalDistribution,
    TestContinuousDistribution,
    TestBetaDistribution,
    TestExponentialDistribution,
    TestDirichletDistribution,
    TestHypergeometricDistribution,
    TestPoisson,
    TestBinomialDistribution,
    TestStudentDistribution,
)

from hela_testing.differential_test import TestDifferential
from hela_testing.antidifferential_test import TestAntriDifferential
from hela_testing.mathfunction_test import (
    TestMathFunctionReal,
    TestPolyvalFunction,
    TestGCD,
    TestModDivision,
)
from hela_testing.nthroot_test import TestNthRootFunction
from hela_testing.pi_function_test import TestPiFunction
from hela_testing.gamma_function_test import TestGammaFunction
from hela_testing.erfc_function_test import TestErfcFunction

if __name__ == "__main__":
    # Create instances of your test classes
    sigmoid_test_suite = unittest.TestLoader().loadTestsFromTestCase(SigmoidTest)
    gaussian_test_suite = unittest.TestLoader().loadTestsFromTestCase(GaussianTest)
    test_normal_distribution_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestNormalDistribution
    )
    test_continous_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestContinuousDistribution
    )
    test_beta_suite = unittest.TestLoader().loadTestsFromTestCase(TestBetaDistribution)
    test_exponential_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestExponentialDistribution
    )
    test_dirichlet_distribution_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDirichletDistribution
    )
    test_relu_suite = unittest.TestLoader().loadTestsFromTestCase(TestReLU)

    test_differential_derivative_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDifferential
    )
    test_binomial_distribution_pmf_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestBinomialDistribution
    )
    test_student_distribution_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestStudentDistribution
    )
    test_hypergeometric_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestHypergeometricDistribution
    )
    test_poisson_suite = unittest.TestLoader().loadTestsFromTestCase(TestPoisson)
    test_bayes_suite = unittest.TestLoader().loadTestsFromTestCase(TestBayesTheorem)

    test_math_function_real_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMathFunctionReal,
        TestPolyvalFunction,
    )

    test_antidifferential_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestAntriDifferential
    )

    test_nthroot_real_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestNthRootFunction
    )

    test_pi_function_suite = unittest.TestLoader().loadTestsFromTestCase(TestPiFunction)

    test_gamma_function_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGammaFunction
    )

    test_erfc_function_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestErfcFunction
    )

    # Combine the test suites
    all_tests = unittest.TestSuite(
        [
            sigmoid_test_suite,
            gaussian_test_suite,
            test_normal_distribution_suite,
            test_continous_suite,
            test_beta_suite,
            test_exponential_suite,
            test_dirichlet_distribution_suite,
            test_differential_derivative_suite,
            test_binomial_distribution_pmf_suite,
            test_student_distribution_suite,
            test_antidifferential_suite,
            test_hypergeometric_suite,
            test_poisson_suite,
            test_bayes_suite,
            test_relu_suite,
            test_math_function_real_suite,
            test_nthroot_real_suite,
            test_pi_function_suite,
            test_gamma_function_suite,
            test_erfc_function_suite,
        ]
    )

    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(all_tests)
