#include "test_check.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Model vocabulary is forbidden everywhere except the adapter layer — the
// documented model boundary. Platform vocabulary is forbidden in the core
// itself; backend directories are the documented vendor boundary and may
// name the platforms they encapsulate. The guard scans headers so the rules
// are enforced rather than reviewed.
int main(int argc, char** argv) {
    GERDOS_CHECK(argc == 2);

    const std::vector<std::string> model_tokens{
        "qwen",
        "llama",
        "deepseek",
        "transformer",
        "attention",
        "kv_cache",
        "kvcache",
        "softmax",
        "mixtral",
        "gemma",
        "bert",
        "gpt",
    };

    const std::vector<std::string> platform_tokens{
        "cuda",
        "rocm",
        "sycl",
        "nvidia",
        "opencl",
        "OpenCL",
        "cl::",
        "clCreate",
        "CL_",
        "hipStream",
        "cuStream",
    };

    std::size_t scanned = 0;

    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(argv[1])) {
        if (!entry.is_regular_file() ||
            entry.path().extension() != ".hpp") {
            continue;
        }

        const auto path = entry.path().string();
        const bool in_core = path.find("/core/") != std::string::npos;
        const bool in_adapters =
            path.find("/adapters/") != std::string::npos;

        std::ifstream input(entry.path());
        GERDOS_CHECK(input.good());

        std::string line;
        while (std::getline(input, line)) {
            // Model vocabulary lives only in the adapter layer — the
            // documented model boundary.
            if (!in_adapters) {
                for (const auto& token : model_tokens) {
                    GERDOS_CHECK(line.find(token) == std::string::npos);
                }
            }

            if (in_core) {
                for (const auto& token : platform_tokens) {
                    GERDOS_CHECK(line.find(token) == std::string::npos);
                }
            }
        }

        ++scanned;
    }

    GERDOS_CHECK(scanned > 0);

    return 0;
}