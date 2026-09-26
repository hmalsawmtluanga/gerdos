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


