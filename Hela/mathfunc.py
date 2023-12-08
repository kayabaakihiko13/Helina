import operator
import math
import cmath
from typing import Union, Optional, Callable, Any, SupportsFloat

INF = 1e300 * 1e300
NINF = -INF
NAN = INF - INF
EPS = 2.2204460492503131e-16
pi = 3.1415926535897932385
e = 2.7182818284590452354
sqrt2 = 1.4142135623730950488
sqrt5 = 2.2360679774997896964
phi = 1.6180339887498948482
ln2 = 0.69314718055994530942
ln10 = 2.302585092994045684
euler = 0.57721566490153286061
catalan = 0.91596559417721901505
khinchin = 2.6854520010653064453
apery = 1.2020569031595942854
logpi = 1.1447298858494001741

_exact_gamma = (
    INF,
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
)
_max_exact_gamma = len(_exact_gamma) - 1
_lanczos_g = 7
_lanczos_p = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

_psi_coeff = [
    0.083333333333333333333,
    -0.0083333333333333333333,
    0.003968253968253968254,
    -0.0041666666666666666667,
    0.0075757575757575757576,
    -0.021092796092796092796,
    0.083333333333333333333,
    -0.44325980392156862745,
    3.0539543302701197438,
    -26.456212121212121212,
]

_erfc_coeff_P = [
    1.0000000161203922312,
    2.1275306946297962644,
    2.2280433377390253297,
    1.4695509105618423961,
    0.66275911699770787537,
    0.20924776504163751585,
    0.045459713768411264339,
    0.0063065951710717791934,
    0.00044560259661560421715,
][::-1]

_erfc_coeff_Q = [
    1.0000000000000000000,
    3.2559100272784894318,
    4.9019435608903239131,
    4.4971472894498014205,
    2.7845640601891186528,
    1.2146026030046904138,
    0.37647108453729465912,
    0.080970149639040548613,
    0.011178148899483545902,
    0.00078981003831980423513,
][::-1]

gauss42 = [
    (0.99839961899006235, 0.0041059986046490839),
    (-0.99839961899006235, 0.0041059986046490839),
    (0.9915772883408609, 0.009536220301748501),
    (-0.9915772883408609, 0.009536220301748501),
    (0.97934250806374812, 0.014922443697357493),
    (-0.97934250806374812, 0.014922443697357493),
    (0.96175936533820439, 0.020227869569052644),
    (-0.96175936533820439, 0.020227869569052644),
    (0.93892355735498811, 0.025422959526113047),
    (-0.93892355735498811, 0.025422959526113047),
    (0.91095972490412735, 0.030479240699603467),
    (-0.91095972490412735, 0.030479240699603467),
    (0.87802056981217269, 0.03536907109759211),
    (-0.87802056981217269, 0.03536907109759211),
    (0.8402859832618168, 0.040065735180692258),
    (-0.8402859832618168, 0.040065735180692258),
    (0.7979620532554873, 0.044543577771965874),
    (-0.7979620532554873, 0.044543577771965874),
    (0.75127993568948048, 0.048778140792803244),
    (-0.75127993568948048, 0.048778140792803244),
    (0.70049459055617114, 0.052746295699174064),
    (-0.70049459055617114, 0.052746295699174064),
    (0.64588338886924779, 0.056426369358018376),
    (-0.64588338886924779, 0.056426369358018376),
    (0.58774459748510932, 0.059798262227586649),
    (-0.58774459748510932, 0.059798262227586649),
    (0.5263957499311922, 0.062843558045002565),
    (-0.5263957499311922, 0.062843558045002565),
    (0.46217191207042191, 0.065545624364908975),
    (-0.46217191207042191, 0.065545624364908975),
    (0.39542385204297503, 0.067889703376521934),
    (-0.39542385204297503, 0.067889703376521934),
    (0.32651612446541151, 0.069862992492594159),
    (-0.32651612446541151, 0.069862992492594159),
    (0.25582507934287907, 0.071454714265170971),
    (-0.25582507934287907, 0.071454714265170971),
    (0.18373680656485453, 0.072656175243804091),
    (-0.18373680656485453, 0.072656175243804091),
    (0.11064502720851986, 0.073460813453467527),
    (-0.11064502720851986, 0.073460813453467527),
    (0.036948943165351772, 0.073864234232172879),
    (-0.036948943165351772, 0.073864234232172879),
]

_zeta_int = [
    -0.5,
    0.0,
    1.6449340668482264365,
    1.2020569031595942854,
    1.0823232337111381915,
    1.0369277551433699263,
    1.0173430619844491397,
    1.0083492773819228268,
    1.0040773561979443394,
    1.0020083928260822144,
    1.0009945751278180853,
    1.0004941886041194646,
    1.0002460865533080483,
    1.0001227133475784891,
    1.0000612481350587048,
    1.0000305882363070205,
    1.0000152822594086519,
    1.0000076371976378998,
    1.0000038172932649998,
    1.0000019082127165539,
    1.0000009539620338728,
    1.0000004769329867878,
    1.0000002384505027277,
    1.0000001192199259653,
    1.0000000596081890513,
    1.0000000298035035147,
    1.0000000149015548284,
]

_zeta_P = [
    -3.50000000087575873,
    -0.701274355654678147,
    -0.0672313458590012612,
    -0.00398731457954257841,
    -0.000160948723019303141,
    -4.67633010038383371e-6,
    -1.02078104417700585e-7,
    -1.68030037095896287e-9,
    -1.85231868742346722e-11,
][::-1]

_zeta_Q = [
    1.00000000000000000,
    -0.936552848762465319,
    -0.0588835413263763741,
    -0.00441498861482948666,
    -0.000143416758067432622,
    -5.10691659585090782e-6,
    -9.58813053268913799e-8,
    -1.72963791443181972e-9,
    -1.83527919681474132e-11,
][::-1]

_zeta_1 = [
    3.03768838606128127e-10,
    -1.21924525236601262e-8,
    2.01201845887608893e-7,
    -1.53917240683468381e-6,
    -5.09890411005967954e-7,
    0.000122464707271619326,
    -0.000905721539353130232,
    -0.00239315326074843037,
    0.084239750013159168,
    0.418938517907442414,
    0.500000001921884009,
]

_zeta_0 = [
    -3.46092485016748794e-10,
    -6.42610089468292485e-9,
    1.76409071536679773e-7,
    -1.47141263991560698e-6,
    -6.38880222546167613e-7,
    0.000122641099800668209,
    -0.000905894913516772796,
    -0.00239303348507992713,
    0.0842396947501199816,
    0.418938533204660256,
    0.500000000000000052,
]

EI_ASYMP_CONVERGENCE_RADIUS = 40.0


def _mathfunction_real(fun_real: Callable, fun_complex: Callable) -> Callable:
    """
    create a function that can handle both real and complex input

    Args:
        fun_real (callable): function takes a real number as input
        fun_complex (callable): function that takes a complex number as inpu

    Returns:
        callable: A function that can handle both real and complex input

    Example:

    >>> def square(x):
    ...     return x**2
    >>> def square_complex(x):
    ...     return x ** 2 + 1j
    >>> math_function = _mathfunction_real(square, square_complex)
    >>> result = math_function(3.0)
    >>> print(result)
    >>> result_complex = math_function(3.0 + 2.0j)
    >>> print(result_complex)
    (7+12j)
    """

    def f(x, **kwargs):
        """
        calculate the result of the appropriate function based on the type of input

        Args:
            x (float or complex): input value
            **kwargs: additional keyword arguments

        Return:
            float or complex: the result of the appropriate function

        Raises:
            ValueError: if the input type is not supported

        Example:
        >>> square = lambda x: x ** 2
        >>> square_complex = lambda x: x ** 2 + 1j
        >>> math_function = _mathfunction_real(square, square_complex)
        >>> result = math_function(3.0)
        >>> print(result)
        9.0
        """
        if isinstance(x, float):
            return fun_real(x)
        if isinstance(x, complex):
            return fun_complex(x)
        try:
            x = float(x)
        except (TypeError, ValueError):
            x = complex(x)
            return fun_complex(x)

    f.__name__ = getattr(fun_real, "__name__", "unknown_function_name")
    return f


def _mathfunction(func_real: Callable, func_complex: Callable) -> Callable:
    """
    create a function that can handle both real and complex input

    Args:
        func_real (callable): function that takes a real number as input
        fun_complex (callable): function that takes a complex number as inpu

    Returns:
        callable: function that can handle both real and complex input

    Example:
    >>> def square(x):
    ...     return x ** 2
    >>> def square_complex(x):
    ...     return x ** 2 + 1j
    >>> math_function = _mathfunction(square, square_complex)
    >>> result = math_function(3.0)
    >>> print(result)
    9.0
    >>> result_complex = math_function(3.0 + 2.0j)
    >>> print(result_complex)
    (7+12j)
    """

    def f(x, **kwargs):
        """
        calculate the result of the appropriate function based on the type of input

        Args:
            x (float or complex): input value
            **kwargs: additional keyword arguments

        Return:
            float or complex: the result of the appropriate function

        Raises:
            ValueError: if the input type is not supported

        Example:
        >>> square = lambda x: x ** 2
        >>> square_complex = lambda x: x ** 2 + 1j
        >>> math_function = _mathfunction_real(square, square_complex)
        >>> result = math_function(3.0)
        >>> print(result)
        9.0
        """
        if isinstance(x, complex):
            return func_complex(x)
        try:
            return func_real(float(x))
        except (TypeError, ValueError) as error:
            print(error)
            return func_complex(complex(x))

    return f


def _mathfunction_n(func_real: Callable, func_complex: Callable) -> Callable:
    """
    create a function that can handle both real and complex input

    Args:
        func_real (callable): take one or more real number as input
        func_complex (callable): take one or more complex number as input

    Return:
        callable: function to handle both (complex and real number)

    Example:
    >>> def square(x):
    ...     return x ** 2
    >>> def square_complex(x):
    ...     return x ** 2 + 1j
    >>> math_function = _mathfunction_n(square, square_complex)
    >>> result = math_function(3.0)
    >>> print(result)
    9.0
    """

    def f(*args, **kwargs):
        """
        calculate reuslt of the appropriate function base on the type of the input

        Args:
            *args: variable number input value
            **kwargs: additional argument

        Return:
            float or complex: result of the appropriate function

        Raises:
            ValueError: if the input is not support

        Example:
        >>> square = lambda x: x ** 2
        >>> square_complex = lambda x: x ** 2 + 1j
        >>> math_function = _mathfunction_n(square, square_complex)
        >>> result = math_function(3.0)
        9.0
        """
        try:
            return func_real(*(float(x) for x in args))
        except (TypeError, ValueError):
            return func_complex(*(complex(x) for x in args))

    f.__name__ = getattr(func_real, "__name__", "unknown_function_name")
    return f


def nthroot(x: float, n: complex) -> float | complex:
    """
    calculate the nth rppt of a number

    Args:
        x (float || complex): number for which calculate the nth root
        n (int): degree of the root

    Return:
        float or complex: the calculate nth root of the number
    """
    r = 1.0 / n
    try:
        return float(x) ** r
    except (ValueError, TypeError):
        return complex(x) ** r


def _sinpi_real(x: float) -> Any:
    """
    calculate the sine of x*pi for real numbers

    Args:
        x (float): the input value
    Returns:
        Any: the sine of x*pi
    """
    if x < 0:
        return -_sinpi_real(-x)
    n, r = divmod(x, 0.5)
    r *= pi
    n %= 4
    if n == 0:
        return math.sin(r)
    if n == 1:
        return math.cos(r)
    if n == 2:
        return -math.sin(r)
    if n == 3:
        return -math.cos(r)


def _cospi_real(x: float) -> Any:
    """
    calculate the cosine of x * pi for real numbers

    Args:
        x (float): the input value
    Returns:
        Any: the cosinue of x*pi
    """
    if x < 0:
        x = -x
    n, r = divmod(x, 0.5)
    r *= pi
    n %= 4
    if n == 0:
        return math.cos(r)
    if n == 1:
        return -math.sin(r)
    if n == 2:
        return -math.cos(r)
    if n == 3:
        return math.sin(r)


def _sinpi_complex(z: complex) -> Any:
    """
    calculate the sine of z*pi for complex number

    Args:
        z (complex): the input value
    Returns:
        Any: the sine of z*pi
    """
    if z.real < 0:
        return -_sinpi_complex(-z)
    n, r = divmod(z.real, 0.5)
    z = pi * complex(r, z.imag)
    n %= 4
    if n == 0:
        return cmath.cos(z)
    if n == 1:
        return -cmath.sin(z)
    if n == 2:
        return -cmath.cos(z)
    if n == 3:
        return cmath.sin(z)


def _cospi_complex(z: complex) -> Any:
    """
    calculate the cosine of z*pi complex numbers

    Args:
        z (complex): the input value
    Returns:
        Any: the cosine z*pi
    """
    if z.real < 0:
        z = -z
    n, r = divmod(z.real, 0.5)
    z = pi * complex(r, z.imag)
    n %= 4
    if n == 0:
        return cmath.cos(z)
    if n == 1:
        return -cmath.sin(z)
    if n == 2:
        return -cmath.cos(z)
    if n == 3:
        return cmath.sin(z)


cospi = _mathfunction_real(_cospi_real, _cospi_complex)
sinpi = _mathfunction_real(_sinpi_real, _sinpi_complex)


def tanpi(x):
    """
    calculate the target of x * pi

    Args:
        x (float): the input value
    Returns:
        float: the tangent of x * pi
    Raises:
        OverflowError: if the result is an overflow

    Example:
    >>> tanpi(0.25)
    1.0
    >>> tanpi(1.5 + 2j)
    1j
    """
    try:
        return sinpi(x) / cospi(x)
    except OverflowError as error:
        print(error)
        if complex(x).imag > 10:
            return 1j
        if complex(x).imag < 10:
            return -1j
        raise


def cotpi(x: float) -> complex:
    """
    calculate the cotangent of x * pi

    Args:
        x (float): the input value
    Returns:
        complex: the contangent of x * pi

    Raises:
        OverflowError: if the result is an overflow

    Example:
    >>> cotpi(0.25)
    1.0j
    >>> cotpi(1.5 + 2j)
    -1j
    """
    try:
        return cospi(x) / sinpi(x)
    except OverflowError:
        if complex(x).imag > 10:
            return -1j
        if complex(x).imag < 10:
            return 1j
        raise


try:
    math.log(-2.0)
except (ValueError, TypeError):
    math_log = math.log
    math_sqrt = math.sqrt


def _gamma_real(x: float) -> Union[float, complex]:
    """
    calculate the gamma function for real numbers

    Args:
        x (float): the input value

    Returns:
        Union[float, complex]: gamma function result

    Raises:
        ZeroDivisionError: if the input is non-positive integer

    Example:
    >>> _gamma_real(5)
    24.0
    >>> _gamma_real(0.5)
    1.77245385091
    >>> _gamma_real(-2)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: gamma function pole
    """
    if (_intx := int(x)) == x:
        if _intx <= 0:
            raise ZeroDivisionError("gamma function pole")
        if _intx <= _max_exact_gamma:
            return _exact_gamma[_intx]
    if x < 0.5:
        return pi / (_sinpi_real(x) * _gamma_real(1 - x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g + 2):
            r += _lanczos_p[i] / (x + i)
        t = x + _lanczos_g + 0.5
        return 2.506628274631000502417 * t ** (x + 0.5) * math.exp(-t) * r


def _gamma_complex(x: float) -> float | complex:
    """
    calculate the gamma function for complex numbers

    Args:
        x (float or complex): the input value

    Returns:
        float or complex: the result of the gamma function

    Example:
    >>> result = _gamma_complex(5)
    >>> print(result)
    (24+0j)
    """
    if not x.imag:
        return complex(_gamma_real(x.real))
    if x.real < 0.5:
        return pi / (_sinpi_complex(x) * _gamma_complex(1 - x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g + 2):
            r += _lanczos_p[i] / (x + i)
        t = x + _lanczos_g + 0.5
        return 2.506628274631000502417 * t ** (x + 0.5) * cmath.exp(-t) * r


pow = _mathfunction_n(operator.pow, lambda x, y: complex(x) ** y)
gamma = _mathfunction_real(_gamma_real, _gamma_complex)
log = _mathfunction_n(math_log, cmath.log)
sqrt = _mathfunction(math_sqrt, cmath.sqrt)
exp = _mathfunction(math.exp, cmath.exp)

cos = _mathfunction_real(math.cos, cmath.cos)
sin = _mathfunction_real(math.sin, cmath.sin)
tan = _mathfunction_real(math.tan, cmath.tan)

acos = _mathfunction(math.acos, cmath.acos)
asin = _mathfunction(math.asin, cmath.asin)
atan = _mathfunction_real(math.atan, cmath.atan)

cosh = _mathfunction_real(math.cosh, cmath.cosh)
sinh = _mathfunction_real(math.sinh, cmath.sinh)
tanh = _mathfunction_real(math.tanh, cmath.tanh)
floor = _mathfunction_real(
    math.floor, lambda z: complex(math.floor(z.real), math.floor(z.imag))
)
ceil = _mathfunction_real(
    math.ceil, lambda z: complex(math.ceil(z.real), math.ceil(z.imag))
)
cos_sin = _mathfunction_real(
    lambda x: (math.cos(x), math.sin(x)), lambda z: (cmath.cos(z), cmath.sin(z))
)
cbrt = _mathfunction(lambda x: x ** (1.0 / 3), lambda z: z ** (1.0 / 3))


def rgamma(x):
    """
    calculate the reciprocal of the gamma function

    Args:
        x (float): the input value

    Returns:
        float: the result of 1/gamma(x)

    Example:
    >>> result = rgamma(4)
    >>> print(result)
    0.16666666666666666
    """
    try:
        return 1.0 / gamma(x)
    except ZeroDivisionError:
        return x * 0.0


def factorial(x):
    """
    calculate the factorial of x

    Args:
        x (float): the input value

    Returns:
        float: the result of gamma(x + 1)

    Example:
    >>> result = factorial(5)
    >>> print(result)
    120.0
    """
    return gamma(x + 1.0)


def arg(x):
    """
    calculate the phase angle of a complex number

    Args:
        x (float or complex): the input value

    Returns:
        float: the phase angle of the complex number

    Example:
    >>> result = arg(3 + 4j)
    >>> print(result)
    0.93
    """
    if isinstance(x, float):
        return math.atan2(0.0, x)
    return math.atan2(x.imag, x.real)


def loggamma(x):
    """
    calculate the natural logarithm of the gamma function

    Args:
        x (float or complex): the input value

    Returns:
        complex: the result of log(gamma(x))

    Example:
    >>> result = loggamma(4)
    >>> print(result)
    2.791165719228053
    """
    if not isinstance(x, (float, complex)):
        try:
            x = float(x)
        except (ValueError, TypeError):
            x = complex(x)
    try:
        xreal = x.real
        ximag = x.imag
    except AttributeError:
        xreal = x
        ximag = 0.0

    if xreal < 0.0:
        if abs(x) < 0.5:
            v = log(gamma(x))
            if ximag == 0:
                v = v.conjugate()
            return v
        z = 1 - x
        try:
            re = z.real
            im = z.imag
        except AttributeError:
            re = z
            im = 0.0
        refloor = floor(re)
        if im == 0.0:
            imsign = 0
        elif im < 0.0:
            imsign = -1
        else:
            imsign = 1
        return (
            (-pi * 1j) * abs(refloor) * (1 - abs(imsign))
            + logpi
            - log(sinpi(z - refloor))
            - loggamma(z)
            + 1j * pi * refloor * imsign
        )
    if x == 1.0 or x == 2.0:
        return x * 0
    p = 0.0
    while abs(x) < 11:
        p -= log(x)
        x += 1.0
    s = 0.918938533204672742 + (x - 0.5) * log(x) - x
    r = 1.0 / x
    r2 = r * r
    s += 0.083333333333333333333 * r
    r *= r2
    s += -0.0027777777777777777778 * r
    r *= r2
    s += 0.00079365079365079365079 * r
    r *= r2
    s += -0.0005952380952380952381 * r
    r *= r2
    s += 0.00084175084175084175084 * r
    r *= r2
    s += -0.0019175269175269175269 * r
    r *= r2
    s += 0.0064102564102564102564 * r
    r *= r2
    s += -0.02955065359477124183 * r
    return s + p


def _polyval(coeffs: list, x: Union[float, complex]) -> Union[float, complex]:
    """
    evaluate a polynomial at a given value

    Args:
        coeffs (list): coefficients of the polynomial in decreasing order of powers
        x (float || complex)

    Example:
    >>> coefficients = [2, 3, 1]
    >>> result = _polyval(coefficients, 2)
    >>> print(result)
    15
    """
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + x * p
    return p


def _erf_taylor(x: float) -> float:
    """
    evaluate the error function using the taylor series

    Args:
        x (float): the input value

    Return:
        float: the result of the error function

    Example:
    >>> _erf_taylor(1.0)
    0.8427007929497148
    """
    x2 = x * x
    s = t = x
    n = 1
    while abs(t) > 1e-17:
        t *= x2 / n
        s -= t / (n + n + 1)
        n += 1
        t *= x2 / n
        s += t / (n + n + 1)
        n += 1
    return 1.1283791670955125739 * s


def _erfc_mid(x: float) -> float:
    """
    evaluate the complementary error function using a polynomial approximation

    Args:
        x (float): the input value

    Returns:
        float: the result of the complementary error function

    Example:
    >>> _erfc_mid(1.0)
    0.15729920705028513
    """
    return exp(-x * x) * _polyval(_erfc_coeff_P, x) / _polyval(_erfc_coeff_Q, x)


def _erfc_asymp(x: float) -> float:
    """
    evaluate the complementary error function using asymptotic series

    Args:
        x (float): the input value
    Returns:
        float: the result of complementary error function

    Example:
    >>> _erfc_asymp(1.0)
    0.15729920705028513
    """
    x2 = x * x
    v = exp(-x2) / x * 0.56418958354775628695
    r = t = 0.5 / x2
    s = 1.0
    for n in range(1, 22, 4):
        s -= t
        t *= r * (n + 2)
        s += t
        t *= r * (n + 4)
        if abs(t) < 1e-17:
            break
    return s * v


def erf(x: float) -> float:
    """
    evaluate the error function

    Args:
        x (float): the input value

    Returns:
        float: the result of the error function

    Example:
    >>> erf(1.0)
    0.8427007929497148
    """
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        return -erf(-x)
    if x >= 1.0:
        if x >= 6.0:
            return 1.0
        return 1.0 - _erfc_mid(x)
    return _erf_taylor(x)


def erfc(x):
    """
    evaluate the complementary error function

    Args:
        x (float): the input value

    Returns:
        float: the result of the complementary error function

    Example:
    >>> erfc(1.0)
    0.15729920705028513
    """
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        if x < -6.0:
            return 2.0
        return 2.0 - erfc(-x)
    if x > 9.0:
        return _erfc_asymp(x)
    if x >= 1.0:
        return _erfc_mid(x)
    return 1.0 - _erf_taylor(x)


# FIXME: while 1
def ei_asymp(z: float, _e1: Optional[bool] = False) -> complex:
    """
    compute the exponential integral using asymptotic expansion

    Args:
        z (float or complex): input value
        _e1 (bool, optional): flag to compute the e1 integral. default is False

    Returns:
        complex: the result of the exponential integral

    Example:
    >>> ei_asymp(2.0)
    (0.42278433509846713-0.17605984310937222j)
    """
    r = 1.0 / z
    s = t = 1.0
    k = 1
    while 1:
        t *= k * r
        s += t
        if abs(t) < 1e-16:
            break
        k += 1
    v = s * exp(z) / z
    if _e1:
        if type(z) is complex:
            zreal = z.real
            zimag = z.imag
        else:
            zreal = z
            zimag = 0.0
        if zimag == 0.0 and zreal > 0.0:
            v += pi * 1j
    else:
        if type(z) is complex:
            if z.imag > 0:
                v += pi * 1j
            if z.imag < 0:
                v -= pi * 1j
    return v


# FIXME: while 1
def ei_taylor(z: Union[complex, float], _e1: Optional[bool] = False) -> complex:
    """
    compute the exponential integral using taylor series

    Args:
        z (float or complex): input value
        _e1 (bool, optional): flag to compute the E1 integral, default is False

    Returns:
        complex: the result of the exponential integral

    Example:
    >>> ei_taylor(2.0)
    (0.42278433509846713-0.17605984310937222j)
    """
    s = t = z
    k = 2
    while 1:
        t = t * z / k
        term = t / k
        if abs(term) < 1e-17:
            break
        s += term
        k += 1
    s += euler
    if _e1:
        s += log(-z)
    else:
        if type(z) is float or z.imag == 0.0:
            s += math_log(abs(z))
        else:
            s += cmath.log(z)
    return s


def ei(z, _e1=False):
    """
    compute the exponential integral

    Args:
        z (float or complex): input value
        _e1 (bool, Optional): flag to compute the E1 integral, default is False

    Returns:
        complex: the result of the exponential integral

    Example:
    >>> ei(2.0)
    (0.42278433509846713-0.17605984310937222j)
    """
    typez = type(z)
    if not isinstance(typez, (float, complex)):
        try:
            z = float(z)
            typez = float
        except (TypeError, ValueError):
            z = complex(z)
            typez = complex
    if not z:
        return -INF
    absz = abs(z)
    if absz > EI_ASYMP_CONVERGENCE_RADIUS:
        return ei_asymp(z, _e1)
    elif absz <= 2.0 or (typez is float and z > 0.0):
        return ei_taylor(z, _e1)
    if typez is complex and z.real > 0.0:
        zref = z / absz
        ref = ei_taylor(zref, _e1)
    else:
        zref = EI_ASYMP_CONVERGENCE_RADIUS * z / absz
        ref = ei_asymp(zref, _e1)
    C = (zref - z) * 0.5
    D = (zref + z) * 0.5
    s = 0.0
    if type(z) is complex:
        _exp = cmath.exp
    else:
        _exp = cmath.exp
    for x, w in gauss42:
        t = C * x + D
        s += w * _exp(t) / t
    ref -= C * s
    return ref


def e1(z):
    """
    compute the exponential integral

    Args:
        z (float or complex): the input value

    Returns:
        complex: the result of the exponential integral E,(x)

    Example:
    >>> result = e1(2.0)
    >>> print(result)
    (0.5741865820328397-0.17605984310937222j)
    """
    typez = type(z)
    if type(z) not in (float, complex):
        try:
            z = float(z)
            typez = float
        except (TypeError, ValueError):
            z = complex(z)
            typez = complex
    if typez is complex and not z.imag:
        z = complex(z.real, 0.0)
    return -ei(-z, _e1=True)


def zeta(s):
    """
    compute the riemann zeta function

    Args:
        s (float or complex): the input value

    Returns:
        Union[float, complex]: the result of the riemann zeta function

    Raises:
        ValueError: if is 1, as it corresponds to a pole

    Example:
    >>> result = zeta(2.0)
    >>> print(result)
    1.6449330668482264
    """
    if not isinstance(s, (float, int)):
        try:
            s = float(s)
        except (ValueError, TypeError):
            try:
                s = complex(s)
                if not s.imag:
                    return complex(zeta(s.real))
            except (ValueError, TypeError):
                pass
            raise NotImplementedError
    if s == 1:
        raise ValueError("zeta(1) pole")
    if s >= 27:
        return 1.0 + 2.0 ** (-s) + 3.0 ** (-s)
    if (n := int(s)) == s:
        if n >= 0:
            return _zeta_int[n]
        if not (n % 2):
            return 0.0
    if s <= 0.0:
        return (
            2.0**s
            * pi ** (s - 1)
            * _sinpi_real(0.5 * s)
            * _gamma_real(1 - s)
            * zeta(1 - s)
        )
    if s <= 2.0:
        if s <= 1.0:
            return _polyval(_zeta_0, s) / (s - 1)
        return _polyval(_zeta_1, s) / (s - 1)
    z = _polyval(_zeta_P, s) / _polyval(_zeta_Q, s)
    return 1.0 + 2.0 ** (-s) + 3.0 ** (-s) + 4.0 ** (-s) * z


def gcd(a: int, b: int) -> Union[int, None]:
    """
    euclid's lemma: d divide a and b, if and only if d divides a-b and b-b
    euclid's algorithm

    Args:
        a (int): first number
        b (int): second number

    Returns:
        (int): the greatest common divisor of a and b

    Raises:
        ValueError: if either a or b is not an integer

    Example:
    >>> gcd(121, 11)
    11
    """
    if isinstance(a, int) and isinstance(b, int):
        try:
            if a < b:
                a, b = b, a

            while a % b != 0:
                a, b = b, a % b
            return b
        except ZeroDivisionError:
            print("cannot division by zero")
    else:
        raise TypeError("parameter a or b must be integer")
    return None


def lcm(a: int, b: int) -> Union[int, None]:
    """
    calculate the least common multiple of two number

    Args:
        a (int): first number
        b (int): second number

    Return:
        (int): the least common multiple of a and b

    Raises:
        ValueError: if either a or b is not an integer
    """
    if isinstance(a, int) and isinstance(b, int):
        try:
            gcd_value = gcd(a, b)
            if gcd_value is not None:
                lcm_value = (a * b) // gcd_value
                return lcm_value
        except ZeroDivisionError:
            print("cannot division by zero")
    else:
        raise ValueError("parameter must be an integer")
    return None


def mod_division(
    a: Union[int, float], b: Union[int, float], n: Union[int, float], precision: int = 0
) -> Union[int, None]:
    """
    modular division

    efficient divide b by a mod n

    GCD (greates common divisor) or HCF (highest common factor)

    given three integers a, b, and n, such that gcd(a, n)=1 and n > 1, the function
    should return an integer x such that 0 <= x <= n - 1 and b /a = x(modn)

    theorem:
    a has multiplicate inverse module n iff gcd(a, n)=1

    Args:
        a (int or float): the divisor
        b (int or float): the divided
        n (int or float): the modulus

    Returns:
        int: the result of b divided by a modulo n

    Raises:
        ValueError; if n is not greater than 1 or if gcd(a, n) not equal 1

    Example:
    >>> mod_division(3, 5, 7)
    1
    """

    def extended_euclid(a, b):
        if b == 0:
            return (1, 0)
        (x, y) = extended_euclid(b, a % b)
        k = a // b
        return (y, x - k * y)

    def invert_mod(a, b):
        (b, x) = extended_euclid(a, n)
        if b < 0:
            b = (b % n + n) % n
        return b

    if (
        not isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and isinstance(n, (int, float))
    ):
        raise TypeError("parameter a, b and n must be integer or float")

    try:
        s = invert_mod(a, n)
        x = (b * s) % n
        return round(x, precision)
    except ZeroDivisionError:
        print("cannot divide by a zero")
        return None


def lucas_number(n_nth_number: int) -> int:
    """
    similiar fibonacci sequence, each lucas number is define sum of its two immediately
    preceding terms, the first two lucase number are 2 and 1, and the subsequent terms are generated
    by adding the previous two numbers
    NOTE: the lucas sequence start with 2 and 1 followed by 3, 4, 7, 11, 18, 29, 47, 76, 123

    Args:
        n_nth_number(int): the nth number
    Return:
        (int): the nth lucase number

    Example:
    >>> lucas_number(20)
    15127
    """
    # using recursive
    if not isinstance(n_nth_number, int):
        raise TypeError(
            "nth number must be positive integer, got {}".format(type(n_nth_number))
        )
    if n_nth_number == 0:
        return 2
    if n_nth_number == 1:
        return 1

    return lucas_number(n_nth_number - 1) + lucas_number(n_nth_number - 2)
