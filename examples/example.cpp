#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#ifdef QD
#include <qd/dd_real.h>
#endif

#include "MPIR/gmres_ir.hpp"
#include "qdldl-luir/luir.hpp"

namespace fs  = std::filesystem;
namespace fmm = fast_matrix_market;

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
    return EXIT_FAILURE;
  }
  const fs::path mtx_path{argv[1]};
  if (!fs::exists(mtx_path)) {
    std::cout << "Error: Matrix file does not exist at " << mtx_path
              << std::endl;
    return EXIT_FAILURE;
  }

  // using UF = float;
  // using UW = double;
  // using UR = dd_real;

  using UF = double;
  using UW = double;
  using UR = double;

  fmm::matrix_market_header header;
  std::vector<std::size_t>  rows, cols;
  std::vector<UW>           vals;

  // Load and sort matrix by column indices
  {
    fmm::read_options options;
    options.generalize_symmetry = false;
    std::ifstream            fin{mtx_path};
    std::vector<std::size_t> rs, cs;
    std::vector<double>      vs;
    // rs and cs are swapped here, matrix read is transposed
    fmm::read_matrix_market_triplet(fin, header, cs, rs, vs, options);
    std::cout << (header.symmetry == fmm::symmetry_type::symmetric
                      ? "symmetric"
                      : "non symmetric")
              << std::endl;
    // Find sort permutation
    std::size_t              n = rs.size();
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::sort(perm, [&](const std::size_t i, const std::size_t j) {
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
    std::ranges::transform(perm, std::back_inserter(rows),
                           [&](const std::size_t i) { return rs[i]; });
    std::ranges::transform(perm, std::back_inserter(cols),
                           [&](const std::size_t i) { return cs[i]; });
    std::ranges::transform(perm, std::back_inserter(vals), [&](std::size_t i) {
      return static_cast<UW>(vs[i]);
    });
  }

  auto                     n   = static_cast<std::size_t>(header.nrows);
  auto                     nnz = static_cast<std::size_t>(header.nnz);
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
  solver.SetTolerance(1e-5);
  solver.SetMaxIRIterations(100);
  solver.SetMaxGmresIterations(50);
  // solver.Compute(Ap, Ai, Ax, 0);
  // std::cout << "factorization complete" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  solver.Compute(Ap, Ai, Ax, 0);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "factorization complete" << std::endl;
  std::cout << "Time taken: " << duration.count() << " ms\n";

  std::vector<UW> x_ref(n);
  for (std::size_t i = 0; i < n; i++) {
    x_ref[i] = static_cast<UW>(1);
  }

  std::vector<UW> b = MatrixMultiply<UW, UW>(Ap, Ai, Ax, x_ref);

  // std::vector<double> Ax_ref(nnz);
  // for (std::size_t i = 0; i < nnz; i++) {
  //   Ax_ref[i] = static_cast<double>(Ax[i]);
  // }
  // std::vector<double> b_ref(n);
  // for (std::size_t i = 0; i < n; i++) {
  //   b_ref[i] = static_cast<double>(b[i]);
  // }
  //
  // std::vector<dd_real> x_ref(n);
  // Stats stats;
  // LUIR_QDLDL<dd_real, qd_real>(n, Ap.data(), Ai.data(), Ax_ref.data(), b_ref.data(), x_ref.data(), &stats);
  // stats.print(mtx_path.c_str());

  start = std::chrono::high_resolution_clock::now();
  std::vector<UW> x = solver.Solve(b);
  end = std::chrono::high_resolution_clock::now();

  // Calculate and print duration in milliseconds
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

  std::cout << "result dnrm2: "
            << Dnrm2<dd_real, UW>(VectorSubtract<UW>(x, x_ref)) /
                   Dnrm2<dd_real, UW>(x_ref)
            << std::endl;
  std::cout << "result infnorm: "
            << InfNrm(VectorSubtract<dd_real>(x, x_ref)) / InfNrm(x_ref)
            << std::endl
            << std::endl;
  std::cout << InfNrm(VectorSubtract<dd_real>(b, MatrixMultiply<dd_real, dd_real>(Ap, Ai, Ax, x))) / InfNrm(b) << std::endl;

  // std::print("x = ");
  // for (std::size_t i = 0; i < n; i++) {
  //   std::cout << x[i] << ", ";
  // }
  // std::cout << std::endl;

  // std::vector<UW> b(n);
  // {
  //   std::ifstream fin{mtx_path.parent_path() /
  //                     fs::path(std::format("input{}.mtx", 0))};
  //   double bi;
  //   for (std::size_t i = 0; i < n; i++) {
  //     fin >> bi;
  //     b[i] = bi;
  //   }
  // }
  // std::vector<qd_real> x_ref(n);
  // {
  //   std::ifstream fin{mtx_path.parent_path() /
  //                     fs::path(std::format("output{}.mtx", 0))};
  //   for (std::size_t i = 0; i < n; i++) {
  //     fin >> x_ref[i];
  //   }
  // }
  //
  // std::vector<UW> x = solver.Solve(b);
  // std::cout << "result dnrm2: "
  //           << Dnrm2<qd_real>(VectorSubtract<qd_real>(x, x_ref)) /
  //                  Dnrm2<qd_real>(x_ref)
  //           << std::endl;
  // std::cout << "result infnorm: "
  //           << InfNrm(VectorSubtract<qd_real>(x, x_ref)) / InfNrm(x_ref)
  //           << std::endl
  //           << std::endl;

  // std::vector<UW> x = solver.Solve(std::vector<UW>(n, 1));
  // GmresLDLIR<std::float64_t, dd_real, qd_real> solver_ref;
  // solver_ref.SetTolerance(1e-40);
  // solver_ref.SetMaxIRIterations(3);
  // solver_ref.Compute(Ap, Ai, Ax);
  // std::vector<dd_real> xref = solver_ref.Solve(std::vector<dd_real>(n, 1));
  //
  // std::cout << "result: "
  //           << InfNrm(VectorSubtract<dd_real>(x, xref)) / InfNrm(xref)
  //           << std::endl;

  // std::vector<UW> xref(n, 1);
  // std::vector<UW> b = MatrixMultiply<UW, UR>(Ap, Ai, Ax, xref);
  // std::vector<UW> x = solver.Solve(b);
  // std::cout << "result: " << InfNrm(VectorSubtract<double>(x, xref))
  //           << std::endl;
}
