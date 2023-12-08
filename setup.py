from setuptools import setup
from os import path
import io
import platform

with open("requirements.txt") as file:
    requirements = file.read().splitlines()

with open("README.md") as readme_file:
    readme = readme_file.read()

info = path.abspath(path.dirname(__file__))
with io.open(path.join(info, "requirements.txt"), encoding="utf-8") as file:
    core_require = file.read().split("\n")
    if platform.system == "windows":
        core_require.append("pywin32")

install_require = [x.strip() for x in core_require if "git+" not in x]

setup(
    name="Hela",
    version="0.0.1-beta",
    description="Curated list math",
    long_description=str(readme),
    url="https://github.com/slowy07/Hela",
    author="arfy slowy",
    packages=["Hela"],
    python_requires=">=3.10",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    install_requires=install_require,
    license="MIT License",
    project_urls={
        "Bug Reports": "https://github.com/slowy07/Hela/issues",
        "Source": "https://github.com/slowy07/Hela",
    },
)
