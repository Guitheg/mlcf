from setuptools import setup

install_requirements = [
    "setuptools==59.5.0",
    "tensorboard",
    "torch-tb-profiler",
    "geneticalgorithm2",
    "freqtrade"
]

setup(
    install_requires=install_requirements,
)
