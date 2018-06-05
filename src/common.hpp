#include <filesystem>
#include <tuple>
#include <vector>

#include <aot.pb.h>

#include <tensor.h>

namespace fs = std::filesystem;

tc::AotBuf readProto(const fs::path &filename);

std::vector<tc::AotBuf> readProtos(const std::vector<fs::path> &filenames);

void writeProto(const fs::path &filename, const tc::AotBuf &proto);

void validate_input_filename(const fs::path &input);
void validate_input_filenames(const std::vector<fs::path> &inputs);
void validate_output_filename(const fs::path &output);

struct OptionsHash {
    size_t operator()(const tc::CudaMappingOptionsProto &opts) const;
};

struct TensorInfoHash {
    size_t operator()(const std::vector<tc::TensorInfo> &tis) const;
};

struct KernelInfoHash {
    size_t operator()(const tc::KernelInfo &ki) const;
};

namespace tc {
bool operator==(const tc::KernelInfo &x, const tc::KernelInfo &y);
} // namespace tc
