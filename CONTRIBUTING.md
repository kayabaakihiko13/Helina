# Contributing guidelines

## Before contributing

Thanks for contributing to Hela. Hela is curated packaged for math computational. it provides a number functions for performing mathematical calculation, so you can create, fixing, and improve hela by following our guidelines, lets go!.

## Contributing

### Contributor

We are very thankfully by you contribution even just ``docs: fix typos``, it was very helpfull but, watch it:

- will be distributed under [MIT License](LICENSE.md) once your pull request is merged.
- Your submitted work fulfills or mostly fulfills our styles and standards.
- have a more information about what are your doing adding a ``commented`` or ``docstring`` and give a understanding commit message

#### Issues

If you are interested in resolving an [open issue](https://github.com/slowy07/Hela/issues), simply make a pull request with your proposed fix.
Please help us keep our issue list small by adding `Fixes #{$ISSUE_NUMBER}` to the description of pull requests that resolve open issues.
For example, if your pull request fixes issue #10, then please add the following to its description:
```
Fixes #10
```
GitHub will use this tag to [auto-close the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if and when the PR is merged.

#### What is an Algorithm?

An Algorithm is one or more functions (or classes) that:
* take one or more inputs,
* perform some internal calculations or data manipulations,
* return one or more outputs,

### Adding your function to testing

If you create a new function or some usefull function, we recommended that you must adding the function to unitesting by

example for the function

```py
def adding_two_numbers(a: int, b: int) -> int:
    """
    create adding of two numbers

    Args:
        a (int): first number
        b (int): second number

    Return:
        (int): result adding first and two numbers
    """
    return a + b
```

if adding to old files like ``mathfunc.py`` (location: [``Hela/mathfunc.py``]9(Hela/mathfunc.py)), you can add into ``mathfunction_test.py`` in ``hela_testing/mathfunction_test.py`` by
```py
import unittest
from Hela.common import mathfunc as mathfunction
class TestTwoNumber(unittest.TestCase):
    def test_function(self):
        self.assertEqual(mathfunction.adding_two_numbers(2, 3), 5)
```

information:
```
self.assertEqual(result from function or class, expectation value)
```

#### Pre-commit plugin
Use [pre-commit](https://pre-commit.com/#installation) to automatically format your code to match our coding style:

```bash
python3 -m pip install pre-commit  # only required the first time
pre-commit install
```
That's it! The plugin will run every time you commit any changes. If there are any errors found during the run, fix them and commit those changes. You can even run the plugin manually on all files:

```bash
pre-commit run --all-files --show-diff-on-failure
```

### Pytest
And dont forget by install [Pytest](https://docs.pytest.org/en/7.4.x/), to testing your code by

```bash
python3 -m pip install pytest
```

after that run the pytest by
```bash
pytest . --verbose
```

### Commit message

We use recommended to use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), cause:
- to standarized format that makes commit message easy to read and understand
- allow developers to quick graps the intent and impact of a commit without having to delve into code itself
- very usefull for automated tools such: bumping version number, anlyzing commit history and trends

for the example
```bash
git commit -m "fix: fixing that function foo and bar cannot return integer"
```


After that thanks for your contributing on Hela, thank you so much !