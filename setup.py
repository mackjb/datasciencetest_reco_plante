from setuptools import setup, find_packages

# Lecture des dépendances depuis requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="datasciencetest-reco-plante",
    version="0.1.0",

    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
)
