#pragma once

// Hardware census: describe the machine without claiming it. CPU thread
// count and RAM bytes are always collected from the OS. OpenCL platforms
// and devices are enumerated only when the toolchain exists
// (GERDOS_HAS_OPENCL, set by CMake); without it the device list stays
// empty. The census never fails and never invents devices — unknown fields
// keep their "(unknown)" / zero sentinels.

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(GERDOS_HAS_OPENCL)
#include <CL/opencl.hpp>
#endif

namespace gerdos {

struct CensusDevice {
    std::string name{"(unknown)"};
    std::string vendor{"(unknown)"};
    std::string version{"(unknown)"};
    std::uint64_t global_mem_bytes{0};
    std::uint64_t max_work_group{0};
    bool is_gpu{false};
    bool is_cpu{false};
};

struct Census {
    unsigned cpu_threads{0};
    std::uint64_t ram_bytes{0};
    std::vector<CensusDevice> opencl_devices;
};

inline std::uint64_t census_ram_bytes() noexcept {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(status.ullTotalPhys);
#else
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(pages) *
           static_cast<std::uint64_t>(page_size);
#endif
}

inline void census_opencl_devices(std::vector<CensusDevice>& out) {
#if defined(GERDOS_HAS_OPENCL)
    std::vector<cl::Platform> platforms;
    if (cl::Platform::get(&platforms) != CL_SUCCESS) {
        return;
    }
    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        if (platform.getDevices(CL_DEVICE_TYPE_ALL, &devices) !=
            CL_SUCCESS) {
            continue;
        }
        for (auto& device : devices) {
            CensusDevice found;
            device.getInfo(CL_DEVICE_NAME, &found.name);
            device.getInfo(CL_DEVICE_VENDOR, &found.vendor);
            device.getInfo(CL_DEVICE_VERSION, &found.version);
            cl_ulong global_mem{0};
            device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem);
            found.global_mem_bytes = static_cast<std::uint64_t>(global_mem);
            std::size_t max_group{0};
            device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_group);
            found.max_work_group = static_cast<std::uint64_t>(max_group);
            cl_device_type kind{0};
            device.getInfo(CL_DEVICE_TYPE, &kind);
            found.is_gpu = (kind & CL_DEVICE_TYPE_GPU) != 0;
            found.is_cpu = (kind & CL_DEVICE_TYPE_CPU) != 0;
            out.push_back(found);
        }
    }
#else
    (void)out;
#endif
}

inline Census collect_census() {
    Census census;
    census.cpu_threads = std::thread::hardware_concurrency();
    census.ram_bytes = census_ram_bytes();
    census_opencl_devices(census.opencl_devices);
    return census;
}

}  // namespace gerdos
