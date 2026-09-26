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

add_executable(gerdos
    src/main.cpp
)

target_include_directories(gerdos
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(gerdos PRIVATE /W4)
else()
    target_compile_options(gerdos PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

enable_testing()

add_executable(gerdos_core_ids_test
    tests/core_ids.cpp
)

target_include_directories(gerdos_core_ids_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(gerdos_core_ids_test PRIVATE /W4)
else()
    target_compile_options(gerdos_core_ids_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(
    NAME gerdos_core_ids
    COMMAND gerdos_core_ids_test
)

add_executable(gerdos_core_operation_test
    tests/core_operation.cpp
)

target_include_directories(gerdos_core_operation_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(gerdos_core_operation_test PRIVATE /W4)
else()
    target_compile_options(gerdos_core_operation_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(
    NAME gerdos_core_operation
    COMMAND gerdos_core_operation_test
)

add_executable(gerdos_core_operation_registry_test
    tests/core_operation_registry.cpp
)

target_include_directories(gerdos_core_operation_registry_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(gerdos_core_operation_registry_test PRIVATE /W4)
else()
    target_compile_options(gerdos_core_operation_registry_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(
    NAME gerdos_core_operation_registry
    COMMAND gerdos_core_operation_registry_test
)

add_executable(gerdos_core_execution_state_test
    tests/core_execution_state.cpp
)

target_include_directories(gerdos_core_execution_state_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(gerdos_core_execution_state_test PRIVATE /W4)
else()
    target_compile_options(gerdos_core_execution_state_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(
    NAME gerdos_core_execution_state
    COMMAND gerdos_core_execution_state_test
)
