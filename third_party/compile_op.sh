#!/bin/sh
# shellcheck disable=SC2164
cd nearest_neighbors
python setup.py build_ext --inplace
cd ../

cd cpp_wrappers
sh compile_wrappers.sh
cd ../../
