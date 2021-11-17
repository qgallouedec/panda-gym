import os

from setuptools import find_packages, setup

with open(os.path.join("panda_gym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="panda_gym",
    description="Set of OpenAI/gym robotic environments based on PyBullet physics engine.",
    author="Quentin GALLOUÉDEC",
    author_email="gallouedec.quentin@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qgallouedec/panda-gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={"panda_gym": ["version.txt"]},
    version=__version__,
    install_requires=["gym", "pybullet", "numpy", "scipy"],
    extras_require={
        "tests": ["pytest", "black", "pytype"],
        "extra": ["numpngw", "stable-baselines3"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
