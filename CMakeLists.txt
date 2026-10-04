cmake_minimum_required(VERSION 3.24)
project(cugpt VERSION 0.1.0 LANGUAGES CXX CUDA)
include(GNUInstallDirs)

# ---------------------------------------------------------
# Build configuration
# ---------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

include(FetchContent)

# ---------------------------------------------------------
# Dependencies
# ---------------------------------------------------------

FetchContent_Declare(
    safetensors_cpp
    GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp
    GIT_TAG main
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.15.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(nanobind)

FetchContent_GetProperties(safetensors_cpp)

if(NOT safetensors_cpp_POPULATED)
    FetchContent_Populate(safetensors_cpp)
endif()

set(_old_skip_install ${CMAKE_SKIP_INSTALL_RULES})
set(CMAKE_SKIP_INSTALL_RULES TRUE)

add_subdirectory(
    ${safetensors_cpp_SOURCE_DIR}
    ${safetensors_cpp_BINARY_DIR}
    EXCLUDE_FROM_ALL
)

set(CMAKE_SKIP_INSTALL_RULES ${_old_skip_install})
unset(_old_skip_install)

# ---------------------------------------------------------
# Deterministic MSVC builds
# ---------------------------------------------------------

if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/experimental:deterministic>
        $<$<COMPILE_LANGUAGE:CXX>:/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=.>

        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/experimental:deterministic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=.>
    )
endif()

# ---------------------------------------------------------
# Core library
# ---------------------------------------------------------

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cu"
)

add_library(cugpt_core STATIC ${CORE_SOURCES})

set_target_properties(cugpt_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(cugpt_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${safetensors_cpp_SOURCE_DIR}"
)

target_link_libraries(cugpt_core PUBLIC
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(cugpt_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

add_executable(cugpt_cli src/main.cpp)

target_link_libraries(cugpt_cli PRIVATE cugpt_core)

set_target_properties(cugpt_cli PROPERTIES
    OUTPUT_NAME cugpt
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------
# Python extension
# ---------------------------------------------------------

nanobind_add_module(cugpt NOMINSIZE
    bindings/cugpt.cpp
)

if(MSVC)
    target_compile_options(nanobind-static PRIVATE /FIcorecrt.h)
    target_compile_options(cugpt PRIVATE /FIcorecrt.h)
endif()

target_link_libraries(cugpt PRIVATE cugpt_core)

set_target_properties(cugpt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------
# Installation into wheel
# ---------------------------------------------------------

install(
    TARGETS cugpt
    LIBRARY DESTINATION cugpt
    RUNTIME DESTINATION cugpt
)

# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

option(CUGPT_BUILD_TESTS "Build cugpt tests" ON)

if(CUGPT_BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
    )

    foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
        get_filename_component(TEST_NAME "${TEST_SOURCE}" NAME_WE)

        add_executable(${TEST_NAME} "${TEST_SOURCE}")

        target_link_libraries(${TEST_NAME} PRIVATE cugpt_core)

        set_target_properties(${TEST_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()