find_package (Eigen3 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
  fast_matrix_market
  GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

add_executable(example ${EXAMPLES_DIR}/example.cpp)
target_link_libraries(example Eigen3::Eigen fast_matrix_market::fast_matrix_market)

# target_include_directories(example PRIVATE ${INCLUDE_DIR} ${QDLDL_INCLUDE_DIR})
# target_link_libraries(example PRIVATE GSL::gsl quadmath)
