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
#include <qd/qd_real.h>
#endif

#ifdef VERBOSE
#include <nlohmann/json.hpp>

// default implementation
template <typename T>
struct TypeName {
  static const char* Get() { return typeid(T).name(); }
};

// a specialization for each type of those you want to support
// and don't like the string returned by typeid
template <>
struct TypeName<float> {
  static const char* Get() { return "single"; }
};
template <>
struct TypeName<double> {
  static const char* Get() { return "double"; }
};
template <>
struct TypeName<std::float16_t> {
  static const char* Get() { return "half"; }
};
template <>
struct TypeName<std::float32_t> {
  static const char* Get() { return "single"; }
};
template <>
struct TypeName<std::float64_t> {
  static const char* Get() { return "double"; }
};

#ifdef QD
template <>
struct TypeName<dd_real> {
  static const char* Get() { return "quadruple"; }
};
template <>
struct TypeName<qd_real> {
  static const char* Get() { return "octuple"; }
};
#endif

#endif

#include "MPIR/gmres_ir.hpp"

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

  using UF = double;
  using UW = double;
  using UR = dd_real;

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
  solver.SetTolerance(1e-15);
  solver.SetMaxIRIterations(200);
  solver.SetMaxGmresIterations(50);
  const std::size_t ILU_K = 0;

  auto start = std::chrono::high_resolution_clock::now();
  solver.Compute<UW, 1 << 21, 5, 2048>(Ap, Ai, Ax, ILU_K);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::println("{:<30}: {} ms", "factorization time", duration.count());

  std::vector<UW> x_ref(n);
  for (std::size_t i = 0; i < n; i++) {
    x_ref[i] = static_cast<UW>(1);
  }

  std::vector<UW> b = MatrixMultiply<UW>(Ap, Ai, Ax, x_ref);

  start             = std::chrono::high_resolution_clock::now();
  std::vector<UW> x = solver.Solve(b);
  end               = std::chrono::high_resolution_clock::now();

  // Calculate and print duration in milliseconds
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::println("{:<30}: {} ms", "solve time", duration.count());

  UW relative_error = InfNrm(VectorSubtract<UW>(x, x_ref)) / InfNrm(x_ref);
  UW relative_residual =
      InfNrm(VectorSubtract<UW>(b, MatrixMultiply<UW>(Ap, Ai, Ax, x))) /
      InfNrm(b);

  std::println("{:<30}: {}", "||x - xref|| / ||xref||", relative_error);
  std::println("{:<30}: {}", "||b - Ax|| / ||b||", relative_residual);

#ifdef VERBOSE
  nlohmann::json    log          = solver.DumpLog();
  const std::string matrix_name  = mtx_path.stem();
  log["matrix"]                  = matrix_name;
  log["factorization_precision"] = TypeName<UF>::Get();
  log["working_precision"]       = TypeName<UW>::Get();
  log["residual_precision"]      = TypeName<UR>::Get();
  log["relative_error"]          = relative_error;
  log["relative_residual"]       = relative_residual;
  log["ilu_k"]                   = ILU_K;
  {
    const auto        now = std::chrono::system_clock::now();
    const std::string timestamp =
        std::format("{:%Y-%m-%d-%H-%M-%S}",
                    std::chrono::zoned_time{std::chrono::current_zone(), now});
    const std::string filename =
        std::format("{}_{}.json", matrix_name, timestamp);
    std::ofstream fout{mtx_path.parent_path() / filename};
    fout << log.dump(2) << std::endl;
  }
#endif

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
}
