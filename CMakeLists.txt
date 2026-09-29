cmake_minimum_required(VERSION 3.24)
project(cugpt LANGUAGES CXX CUDA)

if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/experimental:deterministic>
        $<$<COMPILE_LANGUAGE:CXX>:/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=.>

        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/experimental:deterministic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=.>
    )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(
    safetensors_cpp
    GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp
    GIT_TAG main
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(safetensors_cpp)

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cu"
)

add_library(${PROJECT_NAME}_core ${CORE_SOURCES})

target_include_directories(${PROJECT_NAME}_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${safetensors_cpp_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}_core PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(${PROJECT_NAME}_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(${PROJECT_NAME}_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_core)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})

    target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_NAME}_core)

    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()