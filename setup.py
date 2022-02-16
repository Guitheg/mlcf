from setuptools import setup


if __name__ == "__main__":
    setup(
        dependency_links = [
            "build_helper/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl"
        ],
        install_requires = [
            "ta-lib @ build_helper/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl"
        ]
    )
