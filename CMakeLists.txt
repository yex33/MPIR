cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(MPIR LANGUAGES CXX VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Compiler options
if (WIN32)
    add_compile_options(
            /std:c++latest
            /W4
            $<$<CONFIG:RELEASE>:/O2>
    )
else ()
    add_compile_options(
            -Wall -Wextra -Wpedantic -Wconversion -Wno-sign-conversion -Wno-sign-compare -g
            $<$<CONFIG:DEBUG>:-O0>
            $<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>
            $<$<CONFIG:RELEASE>:-O2>
    )
endif ()

find_library(QD NAMES qd libqd)

# Define the source files and executables
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
set(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
set(TEST_DIR "${PROJECT_SOURCE_DIR}/test")

add_library(MPIR INTERFACE)
target_include_directories(MPIR
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(MPIR
        INTERFACE cxx_std_23
)
if (QD)
    add_compile_definitions(-DQD)
    target_link_libraries(MPIR INTERFACE ${QD})
endif ()

# Installation
include(GNUInstallDirs)

install(
        TARGETS MPIR
        EXPORT MPIRTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
        EXPORT MPIRTargets
        NAMESPACE MPIR::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MPIR
)

enable_testing()
add_subdirectory(test)
add_subdirectory(examples)
