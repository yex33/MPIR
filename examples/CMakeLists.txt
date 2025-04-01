include(FetchContent)
FetchContent_Declare(
  fast_matrix_market
  GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

add_executable(example ${EXAMPLES_DIR}/example.cpp)
target_link_libraries(example PUBLIC
  MPIR
  fast_matrix_market::fast_matrix_market
)
