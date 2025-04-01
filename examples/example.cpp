#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdfloat>
#include <utility>
#include <vector>

#include "gmres_ir.hpp"

namespace fs  = std::filesystem;
namespace fmm = fast_matrix_market;

int main() {
  // const fs::path mtx_path{"../matrices/symmetric/cbuckle.mtx"};
  // const fs::path mtx_path{"../matrices/moderate/2cubes_sphere.mtx"};
  const fs::path mtx_path{"../matrices/symmetric/1138_bus.mtx"};
  if (!fs::exists(mtx_path)) {
    std::cout << "Matrix file does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  using UF = std::float32_t;
  using UW = std::float64_t;
  using UR = std::float64_t;

  fmm::matrix_market_header header;
  std::vector<std::size_t>  rows, cols;
  std::vector<UW>           vals;

  // Load and sort matrix by column indices
  {
    fmm::read_options options;
    options.generalize_symmetry = false;
    std::ifstream            fin{mtx_path};
    std::vector<std::size_t> rs, cs;
    std::vector<UW>          vs;
    // rs and cs are swapped here, matrix read is transposed
    fmm::read_matrix_market_triplet(fin, header, cs, rs, vs, options);
    std::cout << (header.symmetry == fmm::symmetry_type::symmetric
                      ? "symetric"
                      : "non symetric")
              << std::endl;
    // Find sort permutation
    std::size_t              n = rs.size();
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
      if (cs[i] != cs[j]) {
        return cs[i] < cs[j];
      }
      if (rs[i] != rs[j]) {
        return rs[i] < rs[j];
      }
      return false;
    });
    // Apply permutation
    rows.reserve(n);
    cols.reserve(n);
    vals.reserve(n);
    std::transform(perm.begin(), perm.end(), std::back_inserter(rows),
                   [&](std::size_t i) { return rs[i]; });
    std::transform(perm.begin(), perm.end(), std::back_inserter(cols),
                   [&](std::size_t i) { return cs[i]; });
    std::transform(perm.begin(), perm.end(), std::back_inserter(vals),
                   [&](std::size_t i) { return vs[i]; });
  }

  std::size_t              n   = static_cast<std::size_t>(header.nrows);
  std::size_t              nnz = static_cast<std::size_t>(header.nnz);
  std::vector<std::size_t> Ai(std::move(rows));
  std::vector<UW>          Ax(std::move(vals));
  std::vector<std::size_t> Ap(n + 1);
  // Convert COO to CSC. Source:
  // https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
  Ap[0] = 0;
  for (std::size_t i = 0; i < nnz; i++) {
    Ap[cols[i] + 1]++;
  }
  for (std::size_t i = 0; i < n; i++) {
    Ap[i + 1] += Ap[i];
  }

  GmresLDLIR<UF, UW, UR> solver;
  solver.Compute(std::move(Ap), std::move(Ai), std::move(Ax));
  std::vector<UW> b(n, 1);
  solver.Solve(b);
}
