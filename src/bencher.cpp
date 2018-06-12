#include <chrono>
#include <csignal>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>

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



namespace {
DEFINE_string(input, "kernels.proto", "input filename (default: kernels.proto");
DEFINE_uint64(threshold, 50000, "benchmark threshold in us (default: 50000)");
DEFINE_uint64(timelimit, 11, "time limit in hours (default:11)");
} // namespace


auto readProto() {
    if (not fs::exists(FLAGS_input))
        throw std::invalid_argument{FLAGS_input + " does not exists."};

    tc::AotBuf kernels;
    {
        std::ifstream input{FLAGS_input, std::ios::binary};
        if (not kernels.ParseFromIstream(&input))
            throw std::runtime_error{FLAGS_input +
                                     " does not contain a valid protobuf."};
    }
    if (kernels.kernels_size() == 0)
        throw std::runtime_error{"The loaded protobuf is empty."};
    return kernels;
}

void writeProto(const tc::AotBuf &kernels) {
    std::ofstream output{FLAGS_input + "with_runtime",
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
    void operator()(float *t) { cudaFree(t); }
};

struct CudaOutOfMemory : std::exception {};

auto makeTensor(const tc::TensorInfo &ti) {
    float *t;
    auto size =
        std::accumulate(ti.shape.begin(), ti.shape.end(), 1l,
                        [](const auto &a, const auto &b) { return a * b; });
    if (cudaMalloc(&t, size * sizeof(float)) != cudaSuccess) {
        throw CudaOutOfMemory{};
    }

    return std::unique_ptr<float, CudaDeleter>{t, CudaDeleter{}};
}

class TensorCPU {
  public:
    template <typename... SIZES> TensorCPU(SIZES... idx);

    template <typename... IDX> float &operator()(IDX... idx) {
        std::vector<size_t> indices{{idx...}};
        if (indices.size() != info.strides.size()) {
            throw std::invalid_argument{
                "Expected " + std::to_string(info.strides.size()) +
                " indices but got " + std::to_string(indices.size())};
        }
        size_t offset = std::inner_product(indices.begin(), indices.end(),
                                           info.strides.end(), 0ul);
        if (offset >= size) {
            throw std::invalid_argument{
                "Index out of range:  " + std::to_string(offset) +
                ". Size :" + std::to_string(size)};
        }
        return *(memory.get() + offset);
    }

  private:
    std::unique_ptr<float> memory;
    size_t size;
    tc::TensorInfo info;
};

class TensorViewGPU {
  public:
    TensorViewGPU(float *ptr, const tc::TensorInfo &info);
    TensorCPU CPU() const;
    void fromCPU(const TensorCPU &t);

  private:
    float *memory;
    tc::TensorInfo info;
};

class TensorManager {
  public:
    TensorManager() = default;
    TensorManager(const TensorManager &) = delete;
    TensorManager(TensorManager &&) = delete;

    float *getPointerToTensor(const tc::TensorInfo &ti) {
        auto &ts = tensors[ti];
        auto &u = uses[ti];
        if (ts.empty() or u >= ts.size()) {
            ts.push_back(makeTensor(ti));
        }
        ++u;
        return ts.back().get();
    }

    TensorViewGPU getTensorView(const tc::TensorInfo &ti) {
        auto &ts = tensors[ti];
        auto &u = uses[ti];
        if (ts.empty() or u >= ts.size()) {
            ts.push_back(makeTensor(ti));
        }
        ++u;
        return {ts.back().get(), ti};
    }

    void resetUses() {
        for (auto &p : uses) {
            p.second = 0;
        }
    }

    void clear() {
        uses.clear();
        tensors.clear();
    }

  private:
    std::unordered_map<tc::TensorInfo,
                       std::vector<std::unique_ptr<float, CudaDeleter>>,
                       TensorInfoHash>
        tensors;
    std::unordered_map<tc::TensorInfo, size_t, TensorInfoHash> uses;
};

// Halide type handling
typedef int int32;
typedef long int64;
typedef float float32;
typedef double float64;

void group_convolution_4_4_32_56_3_3_32_56(float32 *pO, const float32 *pI,
                                           const float32 *pW1,
                                           const float32 *pB) {
    float32(*O)[32][4][((56 - 3) + 1)][((56 - 3) + 1)] =
        reinterpret_cast<float32(*)[32][4][((56 - 3) + 1)][((56 - 3) + 1)]>(pO);
    const float32(*I)[32][4][56][56] =
        reinterpret_cast<const float32(*)[32][4][56][56]>(pI);
    const float32(*W1)[4][4][3][3] =
        reinterpret_cast<const float32(*)[4][4][3][3]>(pW1);
    const float32(*B)[4] = reinterpret_cast<const float32(*)[4]>(pB);
    for (int c0 = 0; c0 <= 31; c0 += 1) {
        for (int c6 = 0; c6 <= 31; c6 += 1) {
            for (int c7 = 0; c7 <= 3; c7 += 1) {
                for (int c8 = 0; c8 <= 53; c8 += 1) {
                    for (int c9 = 0; c9 <= 53; c9 += 1) {
                        O[c0][c6][c7][c8][c9] = 0.000000f;
                        for (int c10 = 0; c10 <= 3; c10 += 1) {
                            for (int c11 = 0; c11 <= 2; c11 += 1) {
                                for (int c12 = 0; c12 <= 2; c12 += 1) {
                                    O[c0][c6][c7][c8][c9] =
                                        (O[c0][c6][c7][c8][c9] +
                                         (I[c0][c6][c10][(c8 + c11)]
                                           [(c9 + c12)] *
                                          W1[c6][c7][c10][c11][c12]));
                                }
                            }
                        }
                        O[c0][c6][c7][c8][c9] =
                            (O[c0][c6][c7][c8][c9] + B[c6][c7]);
                    }
                }
            }
        }
    }
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

int main(int argc, char *argv[]) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto program_start_time = std::chrono::high_resolution_clock::now();

    static tc::AotBuf kernelBuf;
    kernelBuf = readProto();
    auto kernels = kernelInfoMap(kernelBuf);
    auto deviceName = getDeviceName();

    auto handler = [](int) {
        writeProto(kernelBuf);
        std::abort();
    };
    std::signal(SIGINT, handler);
    std::signal(SIGTERM, handler);
    std::signal(SIGKILL, handler);

    TensorManager tm;
    std::vector<const float *> inputs;
    std::vector<float *> outputs;
    auto number_kernels =
        std::count_if(Register::get().begin(), Register::get().end(),
                      [&kernels](const auto &x) {
                          return kernels.at(x.first)->runtimes_size() == 0;
                      });
    auto benchmarked = 0ul;
    uint64_t total_us = 0;
    uint64_t c = 0;
    for (const auto &p : Register::get()) {
        const auto &id = p.first;
        const auto &kernel_function = p.second;
        const auto &info = kernels.at(id);
        if (info->runtimes_size() > 0) {
            continue;
        }

        while (true) {
            try {
                tm.resetUses();

                inputs.clear();
                outputs.clear();

                for (const auto &i : info->inputs()) {
                    inputs.push_back(tm.getPointerToTensor(tc::TensorInfo{i}));
                }
                for (const auto &o : info->outputs()) {
                    outputs.push_back(tm.getPointerToTensor(tc::TensorInfo{o}));
                }
                tm.resetUses();
                break;
            } catch (const CudaOutOfMemory &e) {
                std::cout
                    << "Out of cuda memory, will free all memory and retry"
                    << std::endl;
                tm.clear();
            }
        }

        uint64_t d = 0;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      kernel_function(inputs, outputs))
                      .count();
        d += us;
        info->add_runtimes(us);
        info->set_device(deviceName);

        if (us < FLAGS_threshold) {
            for (int i = 1; i < 5; ++i) {
                auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                              kernel_function(inputs, outputs))
                              .count();
                d += us;
                info->add_runtimes(us);
            }
        }
        ++benchmarked;
        total_us += d;
        auto us_per_kernel = d / benchmarked;
        auto remaining_us = (number_kernels - benchmarked) * us_per_kernel;
        auto remaining_minutes = remaining_us / 1000 / 1000 / 60;

        std::cout << "Benchmarked " << ++c << "th kernel with id " << id
                  << " : " << d / 5.0f
                  << "us; Estimated Minutes Remaining: " << remaining_minutes
                  << std::endl;
        if (++c % 100 == 0) {
            writeProto(kernelBuf);
        }

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::hours>(now -
                                                           program_start_time)
                .count() >= FLAGS_timelimit) {
            break;
        }
    }
    writeProto(kernelBuf);
}
