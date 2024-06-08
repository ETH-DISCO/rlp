#!/bin/bash

#exit on error
set -e

# create and activate the virtual environment
python3 -m venv rlpvenv
source rlpvenv/bin/activate

# build the C libraries
mkdir -p rlp/lib
cd rlp/lib
cmake ../../puzzles
TMP_MAKEFLAGS=$MAKEFLAGS
export MAKEFLAGS='-j 1'
make icons
export MAKEFLAGS=$TMP_MAKEFLAGS
make
cd ../..

# install rlp and its dependencies
pip install -e .
pip install 'stable_baselines3[extra]>=2.0.0'
pip install 'sb3-contrib>=2.0.0'