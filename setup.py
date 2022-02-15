from setuptools import setup


if __name__ == "__main__":
    setup(
        tests_require=["pytest", 
                    "pytest-mock"],
        install_requires=["torch",
                        "torchvision",
                        "torchaudio",
                        "tensorboard",
                        "torch-tb-profiler",
                        "freqtrade"],
        setup_requires=['flake8']
    )
