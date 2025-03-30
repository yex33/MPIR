#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cstdlib>
#include <fast_matrix_market/app/Eigen.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs  = std::filesystem;
namespace fmm = fast_matrix_market;

int main() {
  const fs::path mtx_path{"matrices/moderate/2cubes_sphere.mtx"};
  if (!fs::exists(mtx_path)) {
    std::cout << "Matrix file does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  std::ifstream fin{mtx_path};

  Eigen::SparseMatrix<double> mat_double;
  fmm::read_matrix_market_eigen(fin, mat_double);
  fin.close();

  Eigen::SparseMatrix<float> mat_single = mat_double.template cast<float>();

  Eigen::VectorXd b = Eigen::VectorXd::Random(mat_single.outerSize(), 1);

  Eigen::SimplicialLDLT<decltype(mat_double)> solver;
  solver.compute(mat_double);
  if (solver.info() != Eigen::Success) {
    std::cout << "decomposition failed" << std::endl;
    return EXIT_FAILURE;
  }
  Eigen::VectorXd x = solver.solve(b);
  if (solver.info() != Eigen::Success) {
    std::cout << "solving step failed" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << x.size() << std::endl;
}
