from setuptools import setup, find_packages

setup(
    name='panda_gym',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'panda_gym': ['envs/assets/*.json'],
    },
    version='0.0.3',
    install_requires=['gym', 'pybullet', 'numpy']
)
