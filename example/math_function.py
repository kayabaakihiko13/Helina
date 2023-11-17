import cmath
from Hela import mathfunc


def square(x):
    return x**2


def square_complex(x):
    return x**2 + 1j


# create a math function using square function
math_square = mathfunc._mathfunction_real(square, square_complex)
math_complex_function = mathfunc._mathfunction_real(square, square_complex)
# test the math function with real number
result_math_square_function = math_square(3.5)
result_math_complex_function = math_complex_function(3.0 + 2.0j)

print(f"result of math square function {result_math_square_function}")
print(f"result math complex function {result_math_complex_function}")
