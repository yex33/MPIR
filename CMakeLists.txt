cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(MPIR LANGUAGES CXX VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compiler options
if(WIN32)
  add_compile_options(
    /std:c++latest
    /W4
    $<$<CONFIG:RELEASE>:/O2>
  )
else()
  add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:DEBUG>:-Og>
    $<$<CONFIG:DEBUG>:-fsanitize=address>
    $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>
    $<$<CONFIG:RELEASE>:-O2>
  )
  add_link_options(
    $<$<CONFIG:DEBUG>:-fsanitize=address>
  )
endif()


# Define the source files and executables
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
set(TEST_DIR "${PROJECT_SOURCE_DIR}/test")
set(QDLDL_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../qdldl-luir-stdloat/include")

add_subdirectory(test)
add_subdirectory(examples)
