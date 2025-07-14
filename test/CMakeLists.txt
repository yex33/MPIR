# doctest
include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
  doctest
  PREFIX ${CMAKE_BINARY_DIR}/doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  TIMEOUT 10
  UPDATE_COMMAND ${GIT_EXECUTABLE} pull
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
)

# Expose required variable (DOCTEST_INCLUDE_DIR) to parent scope
ExternalProject_Get_Property(doctest source_dir)
set(DOCTEST_INCLUDE_DIR ${source_dir}/doctest CACHE INTERNAL "Path to include folder for doctest")

add_executable(doctest_main ${TEST_DIR}/doctest_main.cpp)
target_compile_definitions(doctest_main PRIVATE DOCTEST)
target_include_directories(doctest_main PUBLIC ${DOCTEST_INCLUDE_DIR})
target_link_libraries(doctest_main PRIVATE MPIR)

enable_testing()
add_test(NAME doctests COMMAND $<TARGET_FILE:doctest_main>)
