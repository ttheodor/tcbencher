#include <chrono>
#include <csignal>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <clara.hpp>

#include <aot.pb.h>
#include <benchmark_register.hpp>
#include <tensor.h>

#if __cpp_lib_filesystem >= 201603
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

using namespace clara;

std::string input = "kernels.proto";
uint64_t gpu = 0;

auto cli =
    Opt(input, "input")["--input"]("input filename (default: kernels.proto") |
    Opt(gpu, "gpu")["--gpu"]("which gpu to use (default:0)");

auto readProto() {
    if (not fs::exists(input))
        throw std::invalid_argument{input + " does not exists."};

    tc::AotBuf kernels;
    {
        std::ifstream input_stream{input, std::ios::binary};
        if (not kernels.ParseFromIstream(&input_stream))
            throw std::runtime_error{input +
                                     " does not contain a valid protobuf."};
    }
    if (kernels.kernels_size() == 0)
        throw std::runtime_error{"The loaded protobuf is empty."};
    return kernels;
}

void writeProto(const tc::AotBuf &kernels) {
    std::ofstream output{input + "with_runtime",
                         std::ios::binary | std::ios::trunc};
    if (not kernels.SerializeToOstream(&output))
        throw std::runtime_error{"Failed to serialize protobuf."};
}

auto kernelInfoMap(tc::AotBuf &kernels) {
    std::unordered_map<uint64_t, tc::KernelInfo *> kis;
    for (int i = 0; i < kernels.kernels_size(); ++i) {
        auto *k = kernels.mutable_kernels(i);
        kis[k->id()] = k;
    }
    return kis;
}

struct TensorInfoHash {
    size_t operator()(const tc::TensorInfo &ti) const {
        std::stringstream ss;
        ss << ti.alignment << ',';
        std::copy(ti.shape.begin(), ti.shape.end(),
                  std::ostream_iterator<int64_t>{ss, ","});
        std::copy(ti.strides.begin(), ti.strides.end(),
                  std::ostream_iterator<int64_t>{ss, ","});
        return std::hash<std::string>{}(ss.str());
    }
};
struct CudaDeleter {
    void operator()(float *t) {
        if (cudaFree(t) != cudaSuccess) {
            std::cout << "Could not free cuda memory" << std::endl;
            std::abort();
        }
    }
};

struct CudaOutOfMemory : std::exception {
    CudaOutOfMemory(size_t requested_bytes)
        : requested_bytes{requested_bytes} {}
    size_t requested_bytes;
};

auto makeTensor(const tc::TensorInfo &ti) {
    float *t;
    auto size =
        std::accumulate(ti.shape.begin(), ti.shape.end(), 1l,
                        [](const auto &a, const auto &b) { return a * b; });
    if (cudaMalloc(&t, size * sizeof(float)) != cudaSuccess) {
        throw CudaOutOfMemory{size * sizeof(float)};
    }
    return t;
}

#define Check(condition)                                                       \
    do {                                                                       \
        cudaError_t result = condition;                                        \
        if (result != cudaSuccess) {                                           \
            std::stringstream ss;                                              \
            ss << "Error at: " << __FILE__ << ":" << __LINE__ << ": "          \
               << cudaGetErrorString(result);                                  \
            throw std::runtime_error(ss.str().c_str());                        \
        }                                                                      \
    } while (0)

auto getDeviceName() {
    int deviceID = 0;
    Check(cudaGetDevice(&deviceID));
    cudaDeviceProp deviceProp;
    Check(cudaGetDeviceProperties(&deviceProp, deviceID));
    return std::string{deviceProp.name};
}

struct IOTensors {
    std::vector<const float *> inputs;
    std::vector<float *> outputs;
};

IOTensors allocateCudaTensors(tc::KernelInfo &ki) {
    IOTensors tensors;
    for (auto &it : ki.inputs()) {
        tensors.inputs.push_back(makeTensor(tc::TensorInfo(it)));
    }
    for (auto &ot : ki.outputs()) {
        tensors.outputs.push_back(makeTensor(tc::TensorInfo(ot)));
    }

    return tensors;
}

uint64_t benchKernel(const std::function<Register::KType> &f,
                     IOTensors &tensors) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               f(tensors.inputs, tensors.outputs))
        .count();
}

constexpr int READ = 0;
constexpr int WRITE = 1;

int main(int argc, char *argv[]) {
    auto result = cli.parse(Args(argc, argv));
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage()
                  << std::endl;
        exit(1);
    }
    tc::AotBuf kernelBuf = readProto();
    auto kernels = kernelInfoMap(kernelBuf);

    uint64_t n_benchmarked = 0ul;
    uint64_t total_us = 0ul;

    auto number_kernels =
        std::count_if(Register::get().begin(), Register::get().end(),
                      [&kernels](const auto &x) {
                          return kernels.at(x.first)->runtimes_size() == 0;
                      });

    for (const auto &p : Register::get()) {
        const auto &id = p.first;
        const auto &kernel_function = p.second;
        const auto &info = kernels.at(id);
        if (info->runtimes_size() > 0) {
            continue;
        }
        int pipe_[2];
        if (pipe(pipe_) != 0) {
            std::cout << "Could not create pipe" << std::endl;
            return 1;
        }
        auto cpid = fork();
        if (cpid == -1) {
            std::cout << "Could not fork" << std::endl;
            return 1;
        }

        uint64_t current_median = 0;

        if (cpid == 0) {
            close(pipe_[READ]);
            auto failed = [&]() {
                char s = 1;
                write(pipe_[WRITE], &s, sizeof(s));
            };
            auto success = [&]() {
                char s = 0;
                write(pipe_[WRITE], &s, sizeof(s));
            };
            if (cudaSetDevice(gpu) != cudaSuccess) {
                std::cout << "Could not set gpu device to " << gpu << std::endl;
                failed();
                return 1;
            }

            std::string deviceName;
            try {
                deviceName = getDeviceName();
            } catch (...) {
                std::cout << "Could not get gpu device name." << std::endl;
                failed();
                return 1;
            }

            std::vector<uint64_t> times;
            times.reserve(5);

            try {
                auto tensors = allocateCudaTensors(*info);
                times.push_back(benchKernel(kernel_function, tensors));
                auto iterations = 1;
                if (times.front() < 100000)
                    iterations = 5;
                if (times.front() < 10000)
                    iterations = 11;
                for (int i = 1; i < iterations; ++i) {
                    times.push_back(benchKernel(kernel_function, tensors));
                }
            } catch (...) {
                std::cout << "Could not benchmark kernel." << std::endl;
                failed();
                return 1;
            }

            uint64_t s = times.size();
            success();
            write(pipe_[WRITE], &s, sizeof(s));
            write(pipe_[WRITE], times.data(), s * sizeof(uint64_t));

            uint64_t l = deviceName.length();
            write(pipe_[WRITE], &l, sizeof(l));
            write(pipe_[WRITE], deviceName.data(), l * sizeof(char));
            close(pipe_[WRITE]);
            exit(0);
        } else {
            close(pipe_[WRITE]);
            if (wait(nullptr) == -1) {
                std::cout << "Error waiting for child" << std::endl;
                return 1;
            }

            char status;
            if (read(pipe_[READ], &status, sizeof(status)) != sizeof(status)) {
                std::cout << "Error reading from pipe" << std::endl;
                return 1;
            }
            if (status == 1) {
                std::cout << "Benchmarking kernel with id: " << id << " failed"
                          << std::endl;
                close(pipe_[READ]);
                continue;
            }

            uint64_t s;
            if (read(pipe_[READ], &s, sizeof(s)) != sizeof(s)) {
                std::cout << "Error reading from pipe" << std::endl;
                return 1;
            }
            std::vector<uint64_t> times(s);
            if (read(pipe_[READ], times.data(), s * sizeof(uint64_t)) !=
                s * sizeof(uint64_t)) {
                std::cout << "Error reading from pipe" << std::endl;
                return 1;
            }

            uint64_t l;
            if (read(pipe_[READ], &l, sizeof(l)) != sizeof(l)) {
                std::cout << "Error reading from pipe" << std::endl;
                return 1;
            }

            std::string deviceName(l, '_');
            if (read(pipe_[READ], deviceName.data(), l * sizeof(char)) !=
                l * sizeof(char)) {
                std::cout << "Error reading from pipe" << std::endl;
                return 1;
            }

            close(pipe_[READ]);
            for (auto t : times) {
                info->add_runtimes(t);
                total_us += t;
            }
            info->set_device(deviceName);
            ++n_benchmarked;

            std::nth_element(times.begin(), times.begin() + times.size() / 2,
                             times.end());
            current_median = times.at(times.size() / 2);
        }

        auto us_per_kernel = total_us / n_benchmarked;
        auto remaining_us = (number_kernels - n_benchmarked) * us_per_kernel;
        auto remaining_minutes = remaining_us / 1000 / 1000 / 60;

        std::cout << "Benchmarked " << n_benchmarked << "th out of "
                  << number_kernels << " kernels with id " << id << " : "
                  << current_median
                  << "us; Estimated Minutes Remaining: " << remaining_minutes
                  << std::endl;
    }
    writeProto(kernelBuf);
}
