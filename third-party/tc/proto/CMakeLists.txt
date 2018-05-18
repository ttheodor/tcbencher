find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS mapping_options.proto compcache.proto aot.proto)

add_library(tc_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(tc_proto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(tc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
