#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

#include <aot.pb.h>
#include <clara.hpp>

#include <common.hpp>

using namespace clara;

bool showHelp;

std::vector<fs::path> inputs;
fs::path output;
auto cli = Help(showHelp) |
           Arg(inputs, "input")(
               "The serialized protobufs which containts the kernels.") |
           Opt(output, "output")["--output"](
               "The output directory (must exist be writable.");

namespace tc {
bool operator==(const tc::KernelInfo &x, const tc::KernelInfo &y);
}

class KernelInfoAggregator {
  public:
    void add_proto(const tc::AotBuf &proto) {
        for (const auto &kernel : proto.kernels()) {
            if (kernels.count(kernel) == 0) {
                kernels[kernel] = kernel;
            } else {
                auto &runtimes = *kernels[kernel].mutable_runtimes();
                for (auto r : kernel.runtimes())
                    runtimes.Add(r);
            }
        }
    }

    tc::AotBuf get_merged() const {
        tc::AotBuf buf;
        for (const auto &[key, kernel] : kernels) {
            *buf.add_kernels() = kernel;
        }
        return buf;
    }

  private:
    std::unordered_map<tc::KernelInfo, tc::KernelInfo, KernelInfoHash> kernels;
};

tc::AotBuf merge_protos(const std::vector<tc::AotBuf> &protos) {
    KernelInfoAggregator ka;
    for (const auto &proto : protos)
        ka.add_proto(proto);

    return ka.get_merged();
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

    validate_input_filenames(inputs);
    validate_output_filename(output);

    writeProto(output, merge_protos(readProtos(inputs)));
}
