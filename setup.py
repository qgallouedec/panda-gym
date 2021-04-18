from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="panda_gym",
    description="OpenAI Gym Franka Emika Panda robot environment based on PyBullet.",
    author="Quentin GALLOUÉDEC",
    author_email="gallouedec.quentin@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qgallouedec/panda-gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    version="1.0.1",
    install_requires=["gym", "pybullet", "numpy"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
