#include <chrono>
#include <cuda_runtime.h>
#include <experimental/filesystem>
#include <fstream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>

#include <aot.pb.h>
#include <benchmark_register.hpp>
#include <tensor.h>

namespace {
DEFINE_string(input, "kernels.proto", "input filename (default: kernels.proto");
}

namespace fs = std::experimental::filesystem;

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

auto makeTensor(const tc::TensorInfo &ti) {
    float *t;
    auto size =
        std::accumulate(ti.shape.begin(), ti.shape.end(), 1l,
                        [](const auto &a, const auto &b) { return a * b; });
    if (cudaMalloc(&t, size * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error{"Could not allocate cuda memory."};
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

int main(int argc, char *argv[]) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << "\nDevice name: " << prop.name
                  << std::endl
                  << "Compute capability: " << prop.major << prop.minor
                  << std::endl;
    }

    auto kernelBuf = readProto();
    auto kernels = kernelInfoMap(kernelBuf);

    TensorManager tm;

    std::vector<const float *> inputs;
    std::vector<float *> outputs;
    for (const auto &p : Register::get()) {
        const auto &id = p.first;
        const auto &kernel_function = p.second;

        inputs.clear();
        outputs.clear();

        const auto &info = kernels.at(id);
        for (const auto &i : info->inputs()) {
            inputs.push_back(tm.getPointerToTensor(tc::TensorInfo{i}));
        }
        for (const auto &o : info->outputs()) {
            outputs.push_back(tm.getPointerToTensor(tc::TensorInfo{o}));
        }
        tm.resetUses();

        uint64_t d = 0;
        for (int i = 0; i < 5; ++i) {
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                          kernel_function(inputs, outputs))
                          .count();
            d += us;
            info->add_runtimes(us);
        }
        std::cout << "Benchmarked kernel " << id << " : " << d / 5.0f << "us"
                  << std::endl;
    }
    writeProto(kernelBuf);
}