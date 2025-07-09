from setuptools import setup, find_packages

# Lecture des dépendances depuis requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="datasciencetest-reco-plante",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
)
