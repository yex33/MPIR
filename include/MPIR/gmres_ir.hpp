#pragma once

#ifndef GMRES_IR_HPP
#define GMRES_IR_HPP

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <print>
#include <vector>

#include "fp_concepts.hpp"
#include "ops.hpp"

/**
 * @brief Class implementing GMRES with LDLT-based iterative refinement in
 * mixed-precisions.
 * @tparam UF Factorization precision.
 * @tparam UW Working precision.
 * @tparam UR Residual precision.
 */
template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
class GmresLDLIR {
 private:
  std::size_t              n_ = 0;
  std::vector<std::size_t> Ap_;
  std::vector<std::size_t> Ai_;
  std::vector<UW>          Ax_;
  std::vector<UW>          AD_;

  // U in CSC
  std::vector<std::size_t> Up_;
  std::vector<std::size_t> Ui_;
  std::vector<UF>          Ux_;
  std::vector<UF>          UDinv_;

  // L in CSR
  std::vector<std::size_t> Lp_;
  std::vector<std::size_t> Li_;
  std::vector<UF>          Lx_;

  std::size_t ir_iter_           = 10;
  std::size_t gmres_iter_        = 10;
  UR          tol_               = 1e-10;
  const UW    STAGNATE_THRESHOLD = 1000;

  std::vector<UW> PrecondGmres(const std::vector<UW> &x0,
                               const std::vector<UR> &b);
  std::vector<UW> TriangularSolve(const std::vector<UW> &b);
  UF SparseLDotU(std::size_t row, std::size_t col, std::size_t cut);

 public:
  GmresLDLIR() = default;
  /**
   * @brief Computes and stores the LDLT factorization of the given sparse
   * matrix A.
   * @param Ap Column pointers of A in CSC format.
   * @param Ai Row indices of A in CSC format.
   * @param Ax Non-zero values of A in CSC format.
   * @tparam MAX_MATRIX_SIZE Maximum expected number of rows/columns in the
   * input matrix A. Defaults to 1 << 21. Used to size the bitset for tracking
   * visited row indices efficiently. Ensure this value is greater than the
   * dimension of your largest expected matrix.
   */
  template <typename TAx, std::size_t MAX_MATRIX_SIZE = 1 << 21>
  void Compute(std::vector<std::size_t> Ap,
               std::vector<std::size_t> Ai,
               std::vector<TAx>         Ax,
               std::size_t              IC_k);
  /**
   * @brief Solves the linear system Ax = b using GMRES with LDLT-based
   * refinement.
   * @param b Right-hand side vector.
   * @return Solution vector x.
   */
  std::vector<UW> Solve(const std::vector<UW> &b);

  /**
   * @brief Sets the maximum number of iterative refinement iterations.
   * @param n Number of iterative refinement iterations.
   */
  void SetMaxIRIterations(std::size_t n) { ir_iter_ = n; }

  /**
   * @brief Sets the maximum number of GMRES iterations.
   * @param n Number of GMRES iterations.
   */
  void SetMaxGmresIterations(std::size_t n) { gmres_iter_ = n; }

  /**
   * @brief Sets the convergence tolerance for the solver.
   * @param tol Convergence tolerance value.
   */
  void SetTolerance(UR tol) { tol_ = tol; }
};

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
template <typename TAx, std::size_t MAX_MATRIX_SIZE>
void GmresLDLIR<UF, UW, UR>::Compute(std::vector<std::size_t> Ap,
                                     std::vector<std::size_t> Ai,
                                     std::vector<TAx>         Ax,
                                     const std::size_t        IC_k) {
  // TODO assert Ax.size() <= MAX_MATRIX_SIZE, possible case for solver error
  n_  = Ap.size() - 1;
  Ap_ = std::move(Ap);
  Ai_ = std::move(Ai);
  if constexpr (std::same_as<UW, TAx>) {
    Ax_ = std::move(Ax);
  } else {
    Ax_.resize(Ax.size());
    std::transform(Ax.cbegin(), Ax.cend(), Ax_.begin(),
                   [](TAx xi) { return static_cast<UW>(xi); });
  }

  using std::abs;
  using std::max;
  using std::sqrt;
  // #########################################################################//
  // ##################### Diagonal Scaling of A #############################//
  // #########################################################################//

  // Compute scaling factor
  AD_.resize(n_);
#pragma omp parallel for schedule(static, 256)
  for (std::size_t col = 0; col < n_; col++) {
    for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
      const std::size_t row = Ai_[idx];
      if (row == col && abs(Ax_[idx]) > std::numeric_limits<UF>::epsilon()) {
        AD_[row] = 1.0 / sqrt(abs(Ax_[idx]));
      }
    }
  }
  // Apply scaling
#pragma omp parallel for schedule(static, 256)
  for (std::size_t col = 0; col < n_; ++col) {
    for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
      const std::size_t row = Ai_[idx];
      Ax_[idx] *= AD_[row] * AD_[col];
    }
  }

  // #########################################################################//
  // ##################### Compute Sparsity Pattern ##########################//
  // #########################################################################//

  Up_ = Ap_;
  Ui_ = Ai_;

  for (std::size_t _ = 0; _ < IC_k; _++) {
    std::vector<std::size_t> Up_segments(n_ + 1);
    Up_segments[0] = 0;
    std::vector<std::vector<std::size_t>> Ui_segments(n_);
#pragma omp parallel
    {
      std::bitset<MAX_MATRIX_SIZE> marker;
#pragma omp for schedule(dynamic)
      for (std::size_t col = 0; col < n_; col++) {
        marker.reset();
        std::vector<std::size_t> row_indices;
        row_indices.reserve(Up_[col + 1] - Up_[col]);
        for (std::size_t idx1 = Up_[col]; idx1 < Up_[col + 1]; idx1++) {
          const std::size_t row1 = Ui_[idx1];
          if (row1 > col) continue;
          for (std::size_t idx2 = Ap_[row1]; idx2 < Ap_[row1 + 1]; idx2++) {
            const std::size_t row2 = Ai_[idx2];
            if (row2 > col) continue;
            if (!marker.test(row2)) {
              marker.set(row2);
              row_indices.push_back(row2);
            }
          }
        }
        std::ranges::sort(row_indices);
        Up_segments[col + 1] = row_indices.size();
        Ui_segments[col]     = std::move(row_indices);
      }
    }

    // Exclusive prefix sum of indices
    for (std::size_t col = 1; col <= n_; col++) {
      Up_[col] = Up_[col - 1] + Up_segments[col];
    }

    // Flatten segments
    Ui_.resize(Up_[n_]);
#pragma omp parallel for
    for (std::size_t col = 0; col < n_; col++) {
      std::ranges::copy(Ui_segments[col], std::next(Ui_.begin(), Up_[col]));
    }
  }

  // #########################################################################//
  // ##################### Initial Guesses for U #############################//
  // #########################################################################//

  std::vector<UW> ADnorm(n_);
  ADnorm[0] = 1;
#pragma omp parallel for
  for (std::size_t col = 1; col < n_; col++) {
    auto column =
        std::span<const UW>(Ax_).subspan(Ap_[col], Ap_[col + 1] - Ap_[col] - 1);
    ADnorm[col] = Dnrm2<UW>(column);
    ADnorm[col] = 1;
  }

  Ux_.assign(Ui_.size(), 0);
  UDinv_.resize(n_);
#pragma omp parallel for
  for (std::size_t col = 0; col < n_; col++) {
    std::size_t idx1 = Up_[col], idx2 = Ap_[col];
    while (idx1 < Up_[col + 1] && idx2 < Ap_[col + 1]) {
      if (Ui_[idx1] == Ai_[idx2]) {
        if (Ui_[idx1] == col) {
          // On diagonal
          Ux_[idx1]   = static_cast<UF>(Ax_[idx2]);
          UDinv_[col] = 1.0 / Ux_[idx1];  // Memorize the inverse
        } else {
          // Off diagonal
          Ux_[idx1] = static_cast<UF>(Ax_[idx2] / ADnorm[col]);
        }
        idx1++;
        idx2++;
      } else if (Ui_[idx1] < Ai_[idx2]) {
        idx1++;
      } else {
        idx2++;
      }
    }
  }

  Lp_ = Up_;
  Li_ = Ui_;
  Lx_ = Ux_;

  // #########################################################################//
  // ##################### FGPILU Main Algorithm #############################//
  // #########################################################################//

  UW nonlinear_residual_norm = 0.0;
#pragma omp parallel for reduction(+ : nonlinear_residual_norm) \
    schedule(dynamic, 8)
  for (std::size_t col = 0; col < n_; col++) {
    for (std::size_t idx = Up_[col]; idx < Up_[col + 1]; idx++) {
      const std::size_t row = Ui_[idx];
      // Find A(row, col)
      const UF ax = [this, col, row] {
        for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
          if (Ai_[idx] == row) {
            return static_cast<UF>(Ax_[idx]);
          }
        }
        return static_cast<UF>(0.0);
      }();
      UF sum = SparseLDotU(row, col, row + 1);
      nonlinear_residual_norm += abs(ax - sum);
    }
  }
#pragma omp parallel for reduction(+ : nonlinear_residual_norm) \
    schedule(dynamic, 8)
  for (std::size_t row = 0; row < n_; row++) {
    for (std::size_t idx = Lp_[row]; idx < Lp_[row + 1]; idx++) {
      const std::size_t col = Li_[idx];
      if (col >= row) continue;
      // Find A(row, col)
      const UF ax = [this, row=col, col=row] {
        for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
          if (Ai_[idx] == row) {
            return static_cast<UF>(Ax_[idx]);
          }
        }
        return static_cast<UF>(0.0);
      }();
      UF sum = SparseLDotU(row, col, col + 1);
      nonlinear_residual_norm += abs(ax - sum);
    }
  }
  std::println("Nonlinear residual norm @ 0 sweep is {:g}",
               nonlinear_residual_norm);

  constexpr std::size_t NUM_SWEEPS = 10;
  for (std::size_t sweep = 0; sweep < NUM_SWEEPS; sweep++) {
    std::vector<UF> Ux_new(Ux_.size()), Lx_new(Lx_.size());
    std::vector<UF> UDinv_new(n_);

#pragma omp parallel for schedule(dynamic, 4096)
    for (std::size_t col = 0; col < n_; col++) {
      for (std::size_t idx = Up_[col]; idx < Up_[col + 1]; idx++) {
        const std::size_t row = Ui_[idx];
        // Find A(row, col)
        const UF ax = [this, col, row] {
          for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
            if (Ai_[idx] == row) {
              return static_cast<UF>(Ax_[idx]);
            }
          }
          return static_cast<UF>(0.0);
        }();
        UF sum = SparseLDotU(row, col, row);
        // Fill non-linear solution
        Ux_new[idx] = ax - sum;
        if (row == col) {
          UDinv_new[row] = 1.0 / Ux_new[idx];
        }
      }
    }

#pragma omp parallel for schedule(dynamic, 4096)
    for (std::size_t row = 0; row < n_; row++) {
      for (std::size_t idx = Lp_[row]; idx < Lp_[row + 1]; idx++) {
        const std::size_t col = Li_[idx];
        const UF          ax  = [this, row=col, col=row] {
          for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
            if (Ai_[idx] == row) {
              return static_cast<UF>(Ax_[idx]);
            }
          }
          return static_cast<UF>(0.0);
        }();
        UF sum = SparseLDotU(row, col, col);
        // Fill non-linear solution
        Lx_new[idx] = (ax - sum) * UDinv_[col];
      }
    }

    Ux_    = Ux_new;
    Lx_    = Lx_new;
    UDinv_ = UDinv_new;

    nonlinear_residual_norm = 0.0;
#pragma omp parallel for reduction(+ : nonlinear_residual_norm) \
    schedule(dynamic, 8)
    for (std::size_t col = 0; col < n_; col++) {
      for (std::size_t idx = Up_[col]; idx < Up_[col + 1]; idx++) {
        const std::size_t row = Ui_[idx];
        // Find A(row, col)
        const UF ax = [this, col, row] {
          for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
            if (Ai_[idx] == row) {
              return static_cast<UF>(Ax_[idx]);
            }
          }
          return static_cast<UF>(0.0);
        }();
        UF sum = SparseLDotU(row, col, row + 1);
        nonlinear_residual_norm += abs(ax - sum);
      }
    }
#pragma omp parallel for reduction(+ : nonlinear_residual_norm) \
    schedule(dynamic, 8)
    for (std::size_t row = 0; row < n_; row++) {
      for (std::size_t idx = Lp_[row]; idx < Lp_[row + 1]; idx++) {
        const std::size_t col = Li_[idx];
        if (col >= row) continue;
        // Find A(row, col)
        const UF ax = [this, row=col, col=row] {
          for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
            if (Ai_[idx] == row) {
              return static_cast<UF>(Ax_[idx]);
            }
          }
          return static_cast<UF>(0.0);
        }();
        UF sum = SparseLDotU(row, col, col + 1);
        nonlinear_residual_norm += abs(ax - sum);
      }
    }
    std::println("Nonlinear residual norm @ {} sweep is {:g}", sweep + 1,
                 nonlinear_residual_norm);
    std::println("norm(UDinv) = {:g}", InfNrm(UDinv_));
  }  // End of sweep
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
UF GmresLDLIR<UF, UW, UR>::SparseLDotU(std::size_t row,
                                       std::size_t col,
                                       std::size_t cut) {
  std::size_t idx1 = Lp_[row], idx2 = Up_[col];
  auto        sum = static_cast<UF>(0);
  while (idx1 < Lp_[row + 1] && idx2 < Up_[col + 1]) {
    const std::size_t kL = Li_[idx1], kU = Ui_[idx2];
    if (kL >= cut || kU >= cut) break;
    if (kL == kU) {
      sum += Lx_[idx1] * Ux_[idx2];
      idx1++;
      idx2++;
    } else if (kL < kU) {
      idx1++;
    } else {
      idx2++;
    }
  }
  return sum;
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::Solve(const std::vector<UW> &b) {
  if (ir_iter_ == 0) {
  }
  std::vector<UW> b_scaled = VectorMultiply<UW>(b, AD_);

  std::vector<UW> x(b_scaled);
  std::vector<UR> b0          = MatrixMultiply<UR, UR>(Ap_, Ai_, Ax_, x);
  std::vector<UR> r           = VectorSubtract<UR>(b_scaled, b0);
  UW              d_norm_prev = std::numeric_limits<UW>::max();
  for (std::size_t _ = 0; _ < ir_iter_; _++) {
    UR scale          = InfNrm(r);
    r                 = VectorScale<UR>(r, static_cast<UR>(1) / scale);
    std::vector<UW> d = PrecondGmres({}, r);
    d                 = VectorScale<UW>(d, scale);
    UW d_norm         = InfNrm(d);
    std::cout << "residual " << static_cast<double>(d_norm) << std::endl;
    if (d_norm < tol_ || d_norm / d_norm_prev >= STAGNATE_THRESHOLD) {
      break;
    }
    x           = VectorAdd<UW>(x, d);
    b0          = MatrixMultiply<UR, UR>(Ap_, Ai_, Ax_, x);
    r           = VectorSubtract<UR>(b_scaled, b0);
    d_norm_prev = d_norm;
  }
  return VectorMultiply<UW>(x, AD_);
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::PrecondGmres(const std::vector<UW> &x0,
                                                     const std::vector<UR> &b) {
  if (b.size() != n_) {
    // TODO set solver info
    return {};
  }

  std::vector<UW> res(n_, 0);

  std::vector<UR> r(n_);
  if (x0.empty()) {
    r = b;
  } else {
    std::vector<UR> b0 = MatrixMultiply<UR, UR>(Ap_, Ai_, Ax_, x0);
    r                  = VectorSubtract<UR>(b, b0);
    res                = x0;
  }

  UR rho = Dnrm2<UR>(r);
  if (rho < tol_) {
    return res;
  }

  std::vector<std::vector<UW>> v;
  v.reserve(gmres_iter_ + 1);
  v.emplace_back(n_, 0);
  std::transform(r.cbegin(), r.cend(), v[0].begin(),
                 [](UR ri) { return static_cast<UW>(ri); });
  v[0] = TriangularSolve(v[0]);
  v[0] = VectorScale<UW>(v[0], 1 / rho);

  std::vector<std::vector<UW>> h(gmres_iter_ + 1,
                                 std::vector<UW>(gmres_iter_, 0));
  std::vector<UW>              g(gmres_iter_ + 1, 0);
  std::vector<UW>              c(gmres_iter_, 0);
  std::vector<UW>              s(gmres_iter_, 0);
  g[0] = static_cast<UW>(rho);

  std::size_t k;
  for (k = 0; k < gmres_iter_ && rho > tol_; k++) {
    v.emplace_back(n_, 0);
    v[k + 1] = MatrixMultiply<UW, UR>(Ap_, Ai_, Ax_, v[k]);
    // Apply preconditioner
    v[k + 1] = TriangularSolve(v[k + 1]);

    UW normav = Dnrm2<UW>(v[k + 1]);

    // Modified Gram-Schmidt (MGS)
    for (std::size_t j = 0; j <= k; j++) {
      h[j][k]  = VectorDot<UW>(v[k + 1], v[j]);
      v[k + 1] = VectorSubtract<UW>(v[k + 1], VectorScale<UW>(v[j], h[j][k]));
    }
    UW normav2 = h[k + 1][k] = Dnrm2<UW>(v[k + 1]);

    // Brown/Hindmarsh condition for reorthogonalization
    // if (normav + static_cast<UW>(0.001) * normav2 == normav) {
    //   std::println("reorth triggered at iteration {}", k + 1);
    //   for (std::size_t j = 0; j <= k; j++) {
    //     UW hr = VectorDot<UW>(v[k + 1], v[j]);
    //     h[j][k] += hr;
    //     v[k + 1] = VectorSubtract<UW>(v[k + 1], VectorScale<UW>(v[j], hr));
    //   }
    //   h[k + 1][k] = Dnrm2<UW>(v[k + 1]);
    // }

    // Watch for happy breakdown
    if (h[k + 1][k] != 0) {
      v[k + 1] = VectorScale<UW>(v[k + 1], 1 / h[k + 1][k]);
    }

    // Apply Givens rotation
    for (std::size_t j = 0; j < k; j++) {
      UW w1       = c[j] * h[j][k] - s[j] * h[j + 1][k];
      UW w2       = s[j] * h[j][k] + c[j] * h[j + 1][k];
      h[j][k]     = w1;
      h[j + 1][k] = w2;
    }

    // Form k-th Givens rotation
    UW nu = Dnrm2<UW>(std::vector<UW>{h[k][k], h[k + 1][k]});
    if (nu != 0) {
      c[k] = h[k][k] / nu;
      s[k] = -h[k + 1][k] / nu;

      h[k][k]     = c[k] * h[k][k] - s[k] * h[k + 1][k];
      h[k + 1][k] = 0.0;

      UW w1    = c[k] * g[k] - s[k] * g[k + 1];
      UW w2    = s[k] * g[k] + c[k] * g[k + 1];
      g[k]     = w1;
      g[k + 1] = w2;
    }
    using std::abs;
    rho = abs(g[k + 1]);
  }

  // Solve upper Hessenberg matrix
  std::vector<UW> y(k, 0);
  for (std::size_t i = k; i-- > 0;) {
    y[i] = g[i];
    for (std::size_t j = i + 1; j < k; j++) {
      y[i] -= h[i][j] * y[j];
    }
    y[i] /= h[i][i];
  }

  // Apply correction
  for (std::size_t i = 0; i < n_; i++) {
    for (std::size_t j = 0; j < k; j++) {
      res[i] += v[j][i] * y[j];
    }
  }
  return res;
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::TriangularSolve(
    const std::vector<UW> &b) {
  std::vector<UW> y(n_);

  // Forward solve: L y = b (lower-triangular system)
  for (std::size_t row = 0; row < n_; row++) {
    UW sum = static_cast<UW>(0);
    for (std::size_t idx = Lp_[row]; idx < Lp_[row + 1]; idx++) {
      if (const std::size_t col = Li_[idx]; row != col) {
        sum += Lx_[idx] * y[col];
      }
    }
    y[row] = b[row] - sum;
  }

  std::vector<UW> x = y;
  // Backward solve: U x = y (upper-triangular system)
  for (std::size_t col = n_; col-- > 0;) {
    x[col] *= UDinv_[col];
    if (std::isnan(x[col])) {
      std::println("{}", col);
      exit(1);
    }
    for (std::size_t idx = Up_[col]; idx < Up_[col + 1]; idx++) {
      if (const size_t row = Ui_[idx]; row != col) {
        x[row] -= Ux_[idx] * x[col];
      }
    }
  }
  return x;
}

// End of GMRES_IR_HPP
#endif
