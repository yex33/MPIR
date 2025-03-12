#include <Eigen/Sparse>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <vector>

#include "fast_matrix_market/app/Eigen.hpp"
#include "fast_matrix_market/app/triplet.hpp"

namespace fs  = std::filesystem;
namespace fmm = fast_matrix_market;

struct TripletMatrix {
  size_t              nrows = 0, ncols = 0;
  std::vector<size_t> rows, cols;
  std::vector<double> vals;
};

template <typename U>
struct SparseCsc {
  size_t              nrows = 0, ncols = 0;
  std::vector<size_t> rows;
  std::vector<size_t> col_ptrs;
  std::vector<U>      vals;

  template <typename T>
  SparseCsc(size_t nrows, size_t ncols, std::vector<size_t> rows,
            std::vector<size_t> cols, std::vector<T> vals)
      : nrows(nrows), ncols(ncols), rows(rows) {
    col_ptrs.resize(ncols + 1);
    col_ptrs[0] = 0;
    size_t j    = 0;
    for (size_t i = 0; i < vals.size(); i++) {
    }
  }
};

int main() {
  const fs::path mtx_path{"matrices/moderate/2cubes_sphere.mtx"};
  if (!fs::exists(mtx_path)) {
    std::cout << "Matrix file does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  std::ifstream fin{mtx_path};
  TripletMatrix mat;

  Eigen::SparseMatrix<double> emat;
  fmm::read_matrix_market_eigen(fin, emat);
  emat.makeCompressed();
  std::cout << std::format("outersize = {} innersize = {}\n", emat.outerSize(),
                           emat.innerSize());
  for (size_t i = 0; i < 10; i++) {
    std::cout << std::format("inner index ptr[{}] = {}\n", i,
                             emat.innerIndexPtr()[i]);
  }
  for (size_t i = 0; i < 10; i++) {
    std::cout << std::format("outer index ptr[{}] = {}\n", i,
                             emat.outerIndexPtr()[i]);
  }
  fin.clear();
  fin.seekg(0);
  fmm::read_matrix_market_triplet(fin, mat.nrows, mat.ncols, mat.rows, mat.cols,
                                  mat.vals);

  std::cout << std::format("{} by {} with size {} by {} with val size {}\n",
                           mat.nrows, mat.ncols, mat.rows.size(),
                           mat.cols.size(), mat.vals.size())
            << std::format("first nzz is ({}, {}) = {}\n", mat.rows[0],
                           mat.cols[0], mat.vals[0]);
  fin.close();
}
