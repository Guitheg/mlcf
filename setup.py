from setuptools import setup, find_packages
import mlcf

setup(
    tests_require=["pytest", 
                   "pytest-mock"],
    setup_requires=['flake8']
)
