from setuptools import setup, find_packages
from pathlib import Path

# Lecture des dépendances depuis requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Lire la description depuis le README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="plant-disease-classifier",
    version="1.0.0",
    author="Votre Nom",
    author_email="votre.email@example.com",
    description="Classification des maladies des plantes avec Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-utilisateur/plant-disease-classifier",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    package_dir={"": "."},
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="deep-learning computer-vision plant-disease-classification",
    include_package_data=True,
)
