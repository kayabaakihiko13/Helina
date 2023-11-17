# import matplotlib.pyplot as plt
# import numpy as np
import Hela.mathfunc as mathfunc

result_sin_real = mathfunc._sinpi_real(1.5)
result_cos_complex = mathfunc._cospi_complex(2.5 + 1j)
result_sin_complex = mathfunc._sinpi_complex(1.5 + 2j)
result_cos_real = mathfunc._cospi_real(-3.0)
result_tanpi = mathfunc.tanpi(0.25)
result_tanpi_complex = mathfunc.tanpi(1.5 + 2j)

print(f"sine for real number {result_sin_real}")
print(f"cosine of (2.5 + 1j)*pi for complex numbers: {result_cos_complex}")
print(f"sine of (1.5 + 2j)*pi complex number: {result_sin_complex}")
print(f"cosine -3*pi for real number: {result_cos_real}")
print(f"tanpi pi/4 real number: {result_tanpi}")
print(f"tangent of (1.5 + 2j)* pifor complex number: {result_tanpi_complex}")

# x_values = np.linspace(-5, 5, 400)
# cot_values = [mathfunc.cotpi(mathfunc.pi * x) for x in x_values]
# plt.plot(x_values, np.real(cot_values), label='Real Part')
# plt.plot(x_values, np.imag(cot_values), label='Imaginary Part')
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
# plt.legend()
# plt.title('Cotangent of x*pi')
# plt.xlabel('x')
# plt.ylabel('cot(x*pi)')
# plt.grid(True)
# plt.show()
