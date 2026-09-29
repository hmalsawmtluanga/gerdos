#include "test_check.hpp"

#include <cstdio>

#include "gerdos/hw/census.hpp"

// The census prints what it found, then pins it: CPU/RAM always present,
// OpenCL devices present exactly when the toolchain is compiled in.
int main() {
    const gerdos::Census census = gerdos::collect_census();

    std::printf("cpu threads: %u\n", census.cpu_threads);
    std::printf("ram bytes: %llu\n", (unsigned long long)census.ram_bytes);
    std::printf("cpu name: %s\n", census.cpu_name.c_str());
    std::printf(
        "opencl devices: %u\n",
        (unsigned int)census.opencl_devices.size());
    for (const auto& device : census.opencl_devices) {
        std::printf(
            "opencl: %s | %s | %s | %llu bytes | wg %llu | %s\n",
            device.name.c_str(),
            device.vendor.c_str(),
            device.version.c_str(),
            (unsigned long long)device.global_mem_bytes,
            (unsigned long long)device.max_work_group,
            device.is_gpu ? "GPU" : (device.is_cpu ? "CPU" : "other"));
    }
    std::fflush(stdout);

    GERDOS_CHECK(census.cpu_threads >= 1);
    GERDOS_CHECK(census.ram_bytes > 0);
    GERDOS_CHECK(!census.cpu_name.empty());
#if defined(GERDOS_HAS_OPENCL)
    GERDOS_CHECK(!census.opencl_devices.empty());
    for (const auto& device : census.opencl_devices) {
        GERDOS_CHECK(!device.name.empty());
        GERDOS_CHECK(device.global_mem_bytes > 0);
        GERDOS_CHECK(device.max_work_group > 0);
    }
#else
    GERDOS_CHECK(census.opencl_devices.empty());
#endif
    return 0;
}
