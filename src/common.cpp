#include <common.hpp>

#include <fstream>

#include <boost/functional/hash.hpp>

void writeProto(const std::filesystem::path &filename,
                const tc::AotBuf &proto) {
    std::ofstream output{filename.string(), std::ios::binary | std::ios::trunc};
    if (not proto.SerializeToOstream(&output))
        throw std::runtime_error{"Failed to serialize protobuf."};
}

tc::AotBuf readProto(const std::filesystem::path &filename) {
    tc::AotBuf kernels;
    {
        std::ifstream input{filename.string(), std::ios::binary};
        if (not kernels.ParseFromIstream(&input))
            throw std::runtime_error{filename.string() +
                                     " does not contain a valid protobuf."};
    }
    return kernels;
}

std::vector<tc::AotBuf>
readProtos(const std::vector<std::filesystem::path> &filenames) {
    std::vector<tc::AotBuf> protos;
    protos.reserve(filenames.size());
    std::transform(filenames.begin(), filenames.end(),
                   std::back_inserter(protos), readProto);
    return protos;
}

void validate_output_filename(const fs::path &output) {
    if (output.empty()) {
        throw std::invalid_argument{"The specified output filename is empty."};
    }
    if (fs::exists(output)) {
        throw std::invalid_argument{
            "The specified output file already exists."};
    }
}

void validate_input_filename(const fs::path &input) {
    if (input.empty()) {
        throw std::invalid_argument{"The specified input filename is empty."};
    }
    if (not fs::exists(input)) {
        throw std::invalid_argument{input.string() + " does not exist."};
    }

    if (not fs::is_regular_file(input)) {
        throw std::invalid_argument{input.string() + " is not a regular file."};
    }
}

void validate_input_filenames(const std::vector<fs::path> &inputs) {
    for (const auto &fname : inputs)
        validate_input_filename(fname);
}

namespace tc {
std::size_t hash_value(const tc::TensorInfo &ti) {
    size_t seed = 0;
    boost::hash_combine(seed, ti.dtype.bits);
    boost::hash_combine(seed, ti.dtype.code);
    boost::hash_combine(seed, ti.dtype.lanes);
    boost::hash_combine(seed, ti.alignment);
    for (auto i : ti.shape)
        boost::hash_combine(seed, i);
    for (auto i : ti.strides)
        boost::hash_combine(seed, i);

    return seed;
}
std::size_t hash_value(const tc::TensorInfoProto &ti) {
    size_t seed = 0;
    boost::hash_combine(seed, ti.dtype().bits());
    boost::hash_combine(seed, ti.dtype().code());
    boost::hash_combine(seed, ti.dtype().lanes());
    boost::hash_combine(seed, ti.alignment());
    for (auto i : ti.shape())
        boost::hash_combine(seed, i);
    for (auto i : ti.strides())
        boost::hash_combine(seed, i);

    return seed;
}

std::size_t hash_value(const tc::CudaMappingOptionsProto &opts) {
    return std::hash<std::string>{}(opts.SerializeAsString());
}

std::size_t hash_value(const tc::CudaDimProto &dim) {
    size_t seed = 0;

    boost::hash_combine(seed, dim.x());
    if (dim.has_y())
        boost::hash_combine(seed, dim.y());
    if (dim.has_z())
        boost::hash_combine(seed, dim.z());

    return seed;
}

} // namespace tc

size_t TensorInfoHash::
operator()(const std::vector<tc::TensorInfo> &tis) const {
    size_t seed = 0;
    for (const auto &ti : tis)
        boost::hash_combine(seed, ti);

    return seed;
}

size_t OptionsHash::operator()(const tc::CudaMappingOptionsProto &opts) const {
    return std::hash<std::string>{}(opts.SerializeAsString());
}

size_t KernelInfoHash::operator()(const tc::KernelInfo &ki) const {
    size_t seed = 0;
    boost::hash_combine(seed, ki.tc());
    for (const auto &ti : ki.inputs())
        boost::hash_combine(seed, ti);
    for (const auto &ti : ki.outputs())
        boost::hash_combine(seed, ti);
    boost::hash_combine(seed, ki.kernel_options());
    boost::hash_combine(seed, ki.cuda_source());
    boost::hash_combine(seed, ki.specialized_name());
    for (auto p : ki.parameters())
        boost::hash_combine(seed, p);
    boost::hash_combine(seed, ki.tight_block());
    boost::hash_combine(seed, ki.tight_grid());
    boost::hash_combine(seed, ki.git_version());

    return seed;
}

namespace tc {

bool operator==(const tc::TensorInfoProto &x, const tc::TensorInfoProto &y) {

    if (x.dtype().code() != y.dtype().code())
        return false;
    if (x.dtype().bits() != y.dtype().bits())
        return false;
    if (x.dtype().lanes() != y.dtype().lanes())
        return false;
    if (x.alignment() != y.alignment())
        return false;
    if (x.shape_size() != y.shape_size())
        return false;
    if (x.strides_size() != y.strides_size())
        return false;

    for (int i = 0; i < x.shape_size(); ++i)
        if (x.shape(i) != y.shape(i))
            return false;

    for (int i = 0; i < x.strides_size(); ++i)
        if (x.strides(i) != y.strides(i))
            return false;
    return true;
}

bool operator==(
    const google::protobuf::RepeatedPtrField<tc::TensorInfoProto> &x,
    const google::protobuf::RepeatedPtrField<tc::TensorInfoProto> &y) {
    if (x.size() != y.size())
        return false;
    for (int i = 0; i < x.size(); ++i)
        if (not(x[i] == y[i]))
            return false;
    return true;
}

bool operator==(const tc::KernelInfo &x, const tc::KernelInfo &y) {
    return std::tie(x.tc(), x.inputs()) == std::tie(y.tc(), y.inputs());
}
} // namespace tc
