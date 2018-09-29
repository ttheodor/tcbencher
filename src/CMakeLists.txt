add_subdirectory(cuda_sources)

find_package(Boost REQUIRED)

add_library(tcbencher_lib common.cpp)
target_link_libraries(tcbencher_lib PUBLIC tc)
target_include_directories(tcbencher_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${Boost_INCLUDE_DIRS})

add_executable(bencher bencher.cpp ${cuda_sources})
target_link_libraries(bencher PUBLIC kernels tc stdc++fs clara)
target_include_directories(bencher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_property(TARGET bencher PROPERTY CXX_STANDARD 17)
set_property(TARGET bencher PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(bencher PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bencher PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(generate_cuda generate_cuda.cpp)
target_link_libraries(generate_cuda PUBLIC tc clara stdc++fs)
set_property(TARGET generate_cuda PROPERTY CXX_STANDARD 17)
set_property(TARGET generate_cuda PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


add_executable(proto_merger proto_merger.cpp)
target_link_libraries(proto_merger PUBLIC tcbencher_lib tc clara stdc++fs)
set_property(TARGET proto_merger PROPERTY CXX_STANDARD 17)
set_property(TARGET proto_merger PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
