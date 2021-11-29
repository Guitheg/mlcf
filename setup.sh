#!/usr/bin/env bash
#encoding=utf8

# sudo -i
# git clone https://github.com/freqtrade/freqtrade
# cd freqtrade

# Check which python version is installed
# function check_installed_python() {
#     if [ -n "${VIRTUAL_ENV}" ]; then
#         echo "Please deactivate your virtual environment before running setup.sh."
#         echo "You can do this by running 'deactivate'."
#         exit 2
#     fi

#     for v in 9 8 7
#     do
#         PYTHON="python3.${v}"
#         which $PYTHON
#         if [ $? -eq 0 ]; then
#             echo "using ${PYTHON}"
#             check_installed_pip
#             return
#         fi
#     done

#     echo "No usable python found. Please make sure to have python3.7 or newer installed"
#     exit 1
# }