from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import os

def talib_install():
    dir = "build_helper"
    os.system(f"cd {dir} && sh talib-install.sh")

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
setup(
    cmdclass={"install": TalibInstall,
              "develop": TalibDevelop}
)
