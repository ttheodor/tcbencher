find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS mapping_options.proto compcache.proto aot.proto)

add_library(tc_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(tc_proto PUBLIC /scratch/snx1600/theodort/anaconda/envs/tc_build/lib/libprotobuf.so)
target_include_directories(tc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
