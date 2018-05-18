add_subdirectory(proto)

add_library(tc tensor.cc)
target_include_directories(tc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tc PUBLIC tc_proto)
