add_library(kernels STATIC dummy.cu)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
