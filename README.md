![hela_image_banner](.github/hela.png)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/slowy07/Hela/main.svg)](https://results.pre-commit.ci/latest/github/slowy07/Hela/main)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/slowy07/Hela/pythontest-linux.yml?style=flat-square&logo=python&logoColor=blue&label=Build%20(Linux))
![codefactor grade](https://img.shields.io/codefactor/grade/github/slowy07/Hela?style=flat)
<br/>

Hela is curated packaged for math computational. it provides a number functions for performing mathematical calculations. The function in hela are carefully tested to ensure accuracy and correctness. Hela is great module for who anyone need to perform mathematical calculations with python.


## installation

installation Hela by (only support with python ``3.10``, ``3.11`` and ``3.12`` (for ``3.12`` for experimental))

- clone the repository
```sh
git clone https://github.com/slowy07/hela
```
- install using pip by
```sh
cd hela
pip install .
```

## usage

you can check the simple usage of hela on [`example`](example).

### Differential derivative example

```py
from Hela.common import differential

def (x: float) -> float:
    return x ** 2
value_input: int = 3
calculate_derivative: float = differential.derivative(function, value_input)
print(f"result: {calculate_derivative:.3f}") # 6.000
```

### Contributing

Contributing are welcome, before that you can read the [CONTRIBUTING](CONTRIBUTING.md) before submit pull request.
