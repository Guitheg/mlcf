from setuptools import setup


if __name__ == "__main__":
    setup(
        dependency_links = [
            "build_helper/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl",
            "https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp39-cp39-linux_x86_64.whl"
        ]
    )
