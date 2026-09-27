#include "test_check.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// The core must not depend on vendor APIs or model vocabulary
// (DEVICE_RESOURCE_CONTRACT.md test requirements). This guard scans the core
// headers for forbidden tokens so the rule is enforced rather than reviewed.
int main(int argc, char** argv) {
    GERDOS_CHECK(argc == 2);

    const std::vector<std::string> forbidden{
        "cuda",
        "rocm",
        "sycl",
        "nvidia",
        "hipStream",
        "cuStream",
        "qwen",
        "llama",
        "deepseek",
        "transformer",
        "attention",
        "kv_cache",
        "kvcache",
        "softmax",
    };

    std::size_t scanned = 0;

    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(argv[1])) {
        if (!entry.is_regular_file() ||
            entry.path().extension() != ".hpp") {
            continue;
        }

        std::ifstream input(entry.path());
        GERDOS_CHECK(input.good());

        std::string line;
        while (std::getline(input, line)) {
            for (const auto& token : forbidden) {
                GERDOS_CHECK(line.find(token) == std::string::npos);
            }
        }

        ++scanned;
    }

    GERDOS_CHECK(scanned > 0);

    return 0;
}