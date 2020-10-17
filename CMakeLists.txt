cmake_minimum_required(VERSION 3.18)

project(thrust_benchmark CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

# Where to put build outputs. Use CMAKE_BINARY_DIR so they'll show up in the
# top-level project's dir when building Thrust via add_subdirectory.
set(TBM_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(TBM_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# TODO Allow a specific tag to be fetched with CPM as a CMake option
find_package(Thrust REQUIRED CONFIG)

# TODO Other device systems:
thrust_create_target(ThrustCUDA)
set(THRUST_TARGETS ThrustCUDA)

add_subdirectory(nvbench)
add_subdirectory(benches)
