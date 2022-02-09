import os
from CGrbi.envtools.project import Project, get_dir_prgm


def main():
    project_dir = os.path.join(get_dir_prgm(), "user_data", "Home")
    cgrbi = Project("CGrbi", project_directory=project_dir)
    ### CGrbi ###

    #############
    cgrbi.exit()

if __name__ == "__main__":
    main()