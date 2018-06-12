#!/bin/sh
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=${DIR:=${SOURCE_DIR}}
sources=$(ls ${DIR}/*cu)

printf 'add_library(kernels STATIC '"${sources}"')\ntarget_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)\nset_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)\n' > ${SOURCE_DIR}/CMakeLists.txt
