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
