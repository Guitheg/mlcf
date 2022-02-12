SHELL:=/bin/bash
UNAME:=$(shell uname)
ENV_NAME = ctbt
PYTHON_VERSION:=3.9

CONDA_HOME = $(HOME)/miniconda3
CONDA_BIN_DIR = $(CONDA_HOME)/bin
CONDA = $(CONDA_BIN_DIR)/conda

ENVS_DIR = $(CONDA_HOME)/envs/
ENV_DIR = $(ENVS_DIR)/$(ENV_NAME)
ENV_BIN_DIR = $(ENV_DIR)/bin
ENV_LIB_DIR = $(ENV_DIR)/lib
PYTHON = $(ENV_BIN_DIR)/python
PIP=$(ENV_BIN_DIR)/pip3

setup-run: setup run
setup: add-path conda-install conda-env-create packages-install

####################################################################################################
################################ Add conda to PATH #################################################
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

ifeq (,$(shell echo $(PATH) | grep $(CONDA_BIN_DIR)))
PATH:=$(CONDA_BIN_DIR):$(PATH)
endif
add-path:
ifneq (,$(wildcard $(CONDA_HOME)))
	conda init
endif

####################################################################################################
################# Download and install conda (if it's not installed yet) ###########################
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

ifeq ($(OS),Windows_NT)
CONDASH:=Miniconda3-latest-Windows-x86_64.exe
else
ifeq ($(UNAME),Darwin)
CONDASH:=Miniconda3-latest-MacOSX-x86_64.sh
endif # MacOS
ifeq ($(UNAME),Linux)
CONDASH:=Miniconda3-latest-Linux-x86_64.sh
endif # Linux
endif # not Windows
CONDAURL:=https://repo.anaconda.com/miniconda/$(CONDASH)

conda-install:
ifneq (,$(wildcard $(CONDA_HOME))) # check if conda is installed
	@echo ">>> miniconda3 is installed in $(CONDA_HOME)"
else 
	@echo ">>>$(CONDAURL)"
	@echo ">>> Setting up miniconda3..."
	@echo ">>> Downloading $(CONDAURL) : ... "
	@wget "$(CONDAURL)" && \
	bash "$(CONDASH)" -b -p $(HOME)/miniconda3 && \
	rm -f "$(CONDASH)"
	conda init
endif 

####################################################################################################
#################### Check if the conda environment exist (create if not) ##########################
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
conda-env-create: conda-install
ifneq (,$(wildcard $(ENV_DIR))) # check if the conda environment exist
	@echo ">>> $(ENV_NAME) conda environment exist"
else
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	conda install -n $(ENV_NAME) pip -y
endif

####################################################################################################
################### Check dependencies and update/install packages #################################
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#

packages-install: conda-env-create
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~REQUIREMENTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#---CONDITIONNAL INSTALLATION

#---CONDA INSTALLATION
	conda install -n $(ENV_NAME) -c conda-forge ta-lib -y

#---PIP INSTALLATION
	$(PIP) install -Ur requirements.txt

#---GIT INSTALLATION


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

run:
	$(PYTHON) main.py