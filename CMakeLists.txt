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

# Get Trilinos as one entity
find_package(Trilinos REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

# Echo trilinos build info just for fun
message("\nFound Trilinos!  Here are the details: ")
message("   Trilinos_DIR = ${Trilinos_DIR}")
message("   Trilinos_VERSION = ${Trilinos_VERSION}")
message("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
message("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
message("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
message("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
message("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
message("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
message("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
message("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
message("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
message("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
message("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
message("   Trilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
message("End of Trilinos details\n")

#
# End of setup and error checking

find_library(QD NAMES qd libqd)

# Define the source files and executables
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
set(TEST_DIR "${PROJECT_SOURCE_DIR}/test")

add_library(MPIR INTERFACE)
target_include_directories(MPIR INTERFACE
        "${INCLUDE_DIR}"
)
if (QD)
    add_compile_definitions(-DQD)
    target_link_libraries(MPIR INTERFACE ${QD})
endif ()

enable_testing()
add_subdirectory(test)
add_subdirectory(examples)
