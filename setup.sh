#!/usr/bin/env bash
#encoding=utf8

sudo -i
echo -n "Do you want to install miniconda3 (y/n)? "
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -p $HOME/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    echo "Miniconda3 has been installed"
else
    echo "Miniconda3 has not been installed"
fi

if [ "$answer" != "${answer#[Yy]}" ] ; then
    source $HOME/miniconda3/activate
    echo "Miniconda environment has been activated"
    conda create -n cgrbi python=3.8.8
    conda activate cgrbi
else
    echo "Miniconda environment has not been activated"
else

git clone https://github.com/freqtrade/freqtrade
cd freqtrade
source setup.sh
pip install psutil
pip install freqtrade
cd ..

echo -n "Do you want to keep freqtrade repository (y/n)? "
read fanswer

if [ "$fanswer" != "${fanswer#[Yy]}" ] ; then
    echo "freqtrade repository is keept"
else
    rm -r freqtrade
    echo "freqtrade repository has been deleted"
else

echo "setup succeed"