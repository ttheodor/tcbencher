#include <aot.pb.h>
#include <fstream>
#include <iostream>
#include <string>

#if __cpp_lib_filesystem >= 201603
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <clara.hpp>

using namespace clara;

bool showHelp;
std::string input;
std::string output;
auto cli = Help(showHelp) |
           Opt(input, "input")["--input"](
               "The serialized protobuf which containts the kernels.") |
           Opt(output, "output")["--output"](
               "The output directory (must exist be writable.");

auto readProto(const std::string &filename) {
    if (not fs::exists(filename))
        throw std::invalid_argument{filename + " does not exists."};

    tc::AotBuf kernels;
    {
        std::ifstream input{filename, std::ios::binary};
        if (not kernels.ParseFromIstream(&input))
            throw std::runtime_error{filename +
                                     " does not contain a valid protobuf."};
    }
    if (kernels.kernels_size() == 0)
        throw std::runtime_error{"The loaded protobuf is empty."};
    return kernels;
}

std::string
to_string(const google::protobuf::RepeatedField<google::protobuf::int64> &v) {
    std::stringstream ss;
    for (int i = 0; i < v.size() - 1; ++i) {
        ss << v[i] << ',';
    }
    ss << v[v.size() - 1];
    return ss.str();
}

auto make_params(uint64_t n, const std::string &s) {
    std::stringstream ss;
    for (uint64_t i = 0; i < n - 1; ++i) {
        ss << s << '[' << i << ']' << ',';
    }
    ss << s << '[' << n - 1 << ']';
    return ss.str();
}

auto make_input_params(int n) { return make_params(n, "inputs"); }
auto make_output_params(int n) { return make_params(n, "outputs"); }

static auto preample = R"code(
#include "benchmark_register.hpp"
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
namespace{


#define Check(condition)                                        \
  do {                                                          \
    cudaError_t result = condition;                             \
    if (result != cudaSuccess) {                                \
      std::stringstream ss;                                     \
      ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
         << cudaGetErrorString(result);                         \
      throw std::runtime_error(ss.str().c_str());               \
    }                                                           \
  } while (0))code";

auto makeCode(const tc::KernelInfo &ki) {
    std::stringstream ss;
    ss << preample << '\n'
       << ki.cuda_source() << '\n'
       <<
        R"code(auto dummy = [](){
  Register::get().registerBenchmark([]
  (const std::vector<const float*>& inputs, std::vector<float*>& outputs){
    dim3 grid{)code"
       << ki.tight_grid().x() << ',' << ki.tight_grid().y() << ','
       << ki.tight_grid().z() << "};\n    dim3 block{" << ki.tight_block().x()
       << ',' << ki.tight_block().y() << ',' << ki.tight_block().z() << "};\n"
       << R"code(    cudaEvent_t start, stop;
    Check(cudaEventCreate(&start));
    Check(cudaEventCreate(&stop));
    Check(cudaEventRecord(start, 0));)code"
       << "\n    " << ki.specialized_name() << "<<<grid, block>>>("
       << to_string(ki.parameters()) << ','
       << make_output_params(ki.outputs_size()) << ','
       << make_input_params(ki.inputs_size()) << ')' << ';' <<
        R"code(
    Check(cudaEventRecord(stop, 0));
    Check(cudaEventSynchronize(stop));
    Check(cudaGetLastError());
    float ms = 0.0f;
    Check(cudaEventElapsedTime(&ms, start, stop));
    Check(cudaEventDestroy(start));
    Check(cudaEventDestroy(stop));
    return std::chrono::microseconds{static_cast<uint64_t>(ms*1000)}; }, )code"
       << ki.id() << R"code( );
    return 0;
}();

})code" << '\n';

    return ss.str();
}

int main(int argc, char *argv[]) {
    auto result = cli.parse(Args(argc, argv));
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage()
                  << std::endl;
        exit(1);
    }

    if (showHelp) {
        std::cout << cli;
        exit(1);
    }

    if (input.empty() or output.empty()) {
        std::cerr << "input and/or output were not specified" << std::endl;
        exit(1);
    }

    auto proto = readProto(input);

    auto nextFilename = []() {
        static auto n = 0;
        fs::path p{output};
        std::string fname{"cuda_source_" + std::to_string(n) + ".cu"};
        while (fs::exists(p / fname)) {
            fname = "cuda_source_" + std::to_string(++n) + ".cu";
        }
        return (p / fname).string();
    };

    for (const auto &kernel : proto.kernels()) {
        std::ofstream out{nextFilename()};
        out << makeCode(kernel);
    }
}
