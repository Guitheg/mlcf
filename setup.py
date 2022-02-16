import platform
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import os

WINDOW_OS = "Windows"
LINUX_OS = "Linux"
CURRENT_OS = platform.system()


def talib_install():
    dir = "build_helper"
    if CURRENT_OS == LINUX_OS:
        os.system(f"cd {dir} && sh talib-install.sh")
    elif CURRENT_OS == WINDOW_OS:
        os.system(f"pip install {dir} TA_Lib-0.4.24-cp39-cp39-win_amd64.whl")
    else:
        raise Exception("Unknown OS")


class TalibDevelop(develop):
    def run(self) -> None:
        talib_install()
        return super().run()

    def finalize_options(self) -> None:
        return super().finalize_options()


class TalibInstall(install):
    def run(self):
        talib_install()
        install.run(self)

    def finalize_options(self) -> None:
        return super().finalize_options()


talib_install()

install_requirements = [
    "setuptools==59.5.0",
    "tensorboard",
    "torch-tb-profiler",
    "geneticalgorithm2",
    "ccxt>=1.72.98",
    "freqtrade"
]

setup(
    install_requires=install_requirements,
    cmdclass={"install": TalibInstall,
              "develop": TalibDevelop}
)
