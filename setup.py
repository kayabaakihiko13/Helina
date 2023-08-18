from setuptools import setup

with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="Hela",
    version="0.0.1",
    description="Curated list math",
    author="arfy slowy",
    packages=["Hela"],
    python_requires=">=3.7",
    install_requires=requirements,
)
