cmake_minimum_required(VERSION 3.20)

project(
    GERDOS
    VERSION 0.1.0
    DESCRIPTION "Global Execution Runtime for Dynamic Open Systems"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(gerdos_configure_target target)
    target_include_directories(${target}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()
endfunction()

function(gerdos_add_test test_name source)
    add_executable(${test_name}_test ${source})
    gerdos_configure_target(${test_name}_test)

    add_test(
        NAME ${test_name}
        COMMAND ${test_name}_test
    )
endfunction()

add_executable(gerdos
    src/main.cpp
)

gerdos_configure_target(gerdos)

enable_testing()

gerdos_add_test(gerdos_core_ids
    tests/core_ids.cpp)

gerdos_add_test(gerdos_core_operation
    tests/core_operation.cpp)

gerdos_add_test(gerdos_core_operation_registry
    tests/core_operation_registry.cpp)

gerdos_add_test(gerdos_core_execution_state
    tests/core_execution_state.cpp)

gerdos_add_test(gerdos_core_execution
    tests/core_execution.cpp)

gerdos_add_test(gerdos_core_execution_registry
    tests/core_execution_registry.cpp)

gerdos_add_test(gerdos_core_execution_adversarial
    tests/core_execution_adversarial.cpp)

gerdos_add_test(gerdos_core_physical_binding_resolution
    tests/core_physical_binding_resolution.cpp)

gerdos_add_test(gerdos_core_physical_binding_validation
    tests/core_physical_binding_validation.cpp)

gerdos_add_test(gerdos_core_execution_admission
    tests/core_execution_admission.cpp)

gerdos_add_test(gerdos_core_binding_admissibility
    tests/core_binding_admissibility.cpp)

gerdos_add_test(gerdos_core_execution_effects
    tests/core_execution_effects.cpp)

gerdos_add_test(gerdos_core_execution_backend
    tests/core_execution_backend.cpp)

gerdos_add_test(gerdos_core_simulated_backend
    tests/core_simulated_backend.cpp)

gerdos_add_test(gerdos_core_executor
    tests/core_executor.cpp)

gerdos_add_test(gerdos_core_vertical_slice
    tests/core_vertical_slice.cpp)

gerdos_add_test(gerdos_core_measurement
    tests/core_measurement.cpp)

gerdos_add_test(gerdos_core_binding_planner
    tests/core_binding_planner.cpp)

gerdos_add_test(gerdos_core_workload_benchmark
    tests/core_workload_benchmark.cpp)

gerdos_add_test(gerdos_core_work_description
    tests/core_work_description.cpp)

gerdos_add_test(gerdos_core_work_pool
    tests/core_work_pool.cpp)

gerdos_add_test(gerdos_adapter_model_translation
    tests/adapter_model_translation.cpp)

gerdos_add_test(gerdos_core_cpu_backend
    tests/core_cpu_backend.cpp)

# The heterogeneous backend test requires an OpenCL toolchain and device.
find_package(OpenCL QUIET)
if (NOT OpenCL_FOUND AND DEFINED ENV{GERDOS_OPENCL_ROOT})
    set(OpenCL_INCLUDE_DIRS "$ENV{GERDOS_OPENCL_ROOT}/include")
    find_library(GERDOS_OPENCL_LOADER NAMES libOpenCL.so.1 OpenCL
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib)
    if (GERDOS_OPENCL_LOADER AND
        EXISTS "${OpenCL_INCLUDE_DIRS}/CL/opencl.hpp")
        set(OpenCL_FOUND TRUE)
        set(OpenCL_LIBRARIES "${GERDOS_OPENCL_LOADER}")
    endif()
endif()

if (OpenCL_FOUND)
    gerdos_add_test(gerdos_core_hw_heterogeneous
        tests/core_heterogeneous_backend.cpp)
    target_include_directories(gerdos_core_hw_heterogeneous_test
        PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(gerdos_core_hw_heterogeneous_test
        PRIVATE ${OpenCL_LIBRARIES})
    target_compile_definitions(gerdos_core_hw_heterogeneous_test
        PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300
                CL_HPP_MINIMUM_OPENCL_VERSION=120)

    add_executable(gerdos_demo demo/gerdos_demo.cpp)
    gerdos_configure_target(gerdos_demo)
    target_include_directories(gerdos_demo
        PRIVATE ${OpenCL_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(gerdos_demo PRIVATE ${OpenCL_LIBRARIES})
    target_compile_definitions(gerdos_demo
        PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300
                CL_HPP_MINIMUM_OPENCL_VERSION=120)
    add_test(NAME demo_end_to_end COMMAND gerdos_demo)

    gerdos_add_test(gerdos_core_hw_real_benchmark
        tests/core_real_benchmark.cpp)
    target_include_directories(gerdos_core_hw_real_benchmark_test
        PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(gerdos_core_hw_real_benchmark_test
        PRIVATE ${OpenCL_LIBRARIES})
    target_compile_definitions(gerdos_core_hw_real_benchmark_test
        PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300
                CL_HPP_MINIMUM_OPENCL_VERSION=120)
else()
    message(STATUS
        "OpenCL not found: the heterogeneous backend test is not registered")
endif()

# The Vulkan compute backend test requires a Vulkan toolchain and a real
# GPU with a compute queue (software rasterizers do not count). The test
# registers only when the toolchain is found and aborts at vk_available()
# on machines without a suitable device — the established hardware-gating
# pattern.
find_package(Vulkan QUIET)

if (Vulkan_FOUND)
    gerdos_add_test(gerdos_core_vk_compute
        tests/core_vk_compute_backend.cpp)
    target_include_directories(gerdos_core_vk_compute_test
        PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(gerdos_core_vk_compute_test
        PRIVATE ${Vulkan_LIBRARIES})
else()
    message(STATUS
        "Vulkan not found: the Vulkan compute backend test is not registered")
endif()

add_executable(gerdos_core_no_leakage_test
    tests/core_no_leakage.cpp
)

gerdos_configure_target(gerdos_core_no_leakage_test)

add_test(
    NAME gerdos_core_no_leakage
    COMMAND gerdos_core_no_leakage_test ${CMAKE_CURRENT_SOURCE_DIR}/include
)


