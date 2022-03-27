import os
import sys


def main():
    """Build the documentation thanks to sphinx from the docstring in the source code"""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sphinx_doc_dir = f"{curr_dir}/source"
    project_src_dir = f"{curr_dir}/../../mlcf"
    os.system(f"sphinx-apidoc -f -e -o {sphinx_doc_dir} {project_src_dir}")
    os.system(f"cd {curr_dir} && make html")
    sys.exit(0)


if __name__ == "__main__":
    main()
