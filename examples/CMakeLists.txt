find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP)

# set(MKL_ARCH intel64)
# set(MKL_THREADING sequential)
# set(MKL_INTERFACE lp64)
# set(MKL_DIR "/opt/intel/oneapi/mkl")
# find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
# message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

include(FetchContent)
FetchContent_Declare(
  fast_matrix_market
  GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

add_executable(example ${EXAMPLES_DIR}/example.cpp)
# target_compile_options(example
#   PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
# )
# target_include_directories(example
#   PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
# )
target_link_libraries(example PUBLIC
  # PUBLIC $<LINK_ONLY:MKL::MKL>
  MPIR
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  fast_matrix_market::fast_matrix_market
)
# target_link_libraries(example PRIVATE GSL::gsl quadmath)
