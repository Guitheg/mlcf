from setuptools import setup
from setuptools.command.install import install
import os

class TalibInstall(install):
    def run(self):
        os.system("sh build_helper/talib-install.sh")
        install.run(self)
        
    def finalize_options(self) -> None:
        return super().finalize_options()


setup(
    cmdclass={'install': TalibInstall}
)
