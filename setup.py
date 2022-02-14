from setuptools import setup, find_packages
import mlcf

setup(
    tests_require=["pytest", 
                   "pytest-mock"],
    install_requires=["freqtrade",
                      "torch==1.10.2+cu113",
                      "tensorboard",
                      "torch-tb-profiler"],
    setup_requires=['flake8']
)
