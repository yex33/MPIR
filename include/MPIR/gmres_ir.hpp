#pragma once

#ifndef GMRES_IR_HPP
#define GMRES_IR_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "fp_concepts.hpp"
#include "ops.hpp"
#include "qdldl.hpp"

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

  std::vector<std::size_t> Lp_;
  std::vector<std::size_t> Li_;
  std::vector<UF>          Lx_;
  std::vector<UF>          D_;
  std::vector<UF>          Dinv_;

  std::size_t ir_iter_           = 10;
  std::size_t gmres_iter_        = 10;
  UR          tol_               = 1e-10;
  const UW    STAGNATE_THRESHOLD = 0.5;

  std::vector<UW> PrecondGmres(const std::vector<UW> &x0,
                               const std::vector<UR> &b);

 public:
  GmresLDLIR() = default;
  /**
   * @brief Computes and stores the LDLT factorization of the given sparse
   * matrix A.
   * @param Ap Column pointers of A in CSC format.
   * @param Ai Row indices of A in CSC format.
   * @param Ax Non-zero values of A in CSC format.
   */
  template <typename TAx>
  void Compute(std::vector<std::size_t> Ap,
               std::vector<std::size_t> Ai,
               std::vector<TAx>         Ax);
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
template <typename TAx>
void GmresLDLIR<UF, UW, UR>::Compute(std::vector<std::size_t> Ap,
                                     std::vector<std::size_t> Ai,
                                     std::vector<TAx>         Ax) {
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

  std::vector<std::size_t> iwork(3 * n_);
  std::vector<std::size_t> Lcolnz(n_);
  std::vector<std::size_t> etree(n_);

  std::size_t Lnz = QDLDL_etree(n_, Ap_.data(), Ai_.data(), iwork.data(),
                                Lcolnz.data(), etree.data());

  Lp_.resize(n_ + 1);
  Li_.resize(Lnz);
  Lx_.resize(Lnz);
  D_.resize(n_);
  Dinv_.resize(n_);

  bool           *bwork = new bool[n_];
  std::vector<UF> fwork(n_);
  if constexpr (std::same_as<UF, UW>) {
    QDLDL_factor(n_, Ap_.data(), Ai_.data(), Ax_.data(), Lp_.data(), Li_.data(),
                 Lx_.data(), D_.data(), Dinv_.data(), Lcolnz.data(),
                 etree.data(), bwork, iwork.data(), fwork.data());
  } else {
    std::vector<UF> Ax_UF(Ax_.size());
    std::transform(Ax_.cbegin(), Ax_.cend(), Ax_UF.begin(),
                   [](UW x) { return static_cast<UF>(x); });
    QDLDL_factor(n_, Ap_.data(), Ai_.data(), Ax_UF.data(), Lp_.data(),
                 Li_.data(), Lx_.data(), D_.data(), Dinv_.data(), Lcolnz.data(),
                 etree.data(), bwork, iwork.data(), fwork.data());
  }
  delete[] bwork;
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::Solve(const std::vector<UW> &b) {
  if (ir_iter_ == 0) {
  }
  std::vector<UW> x(b);
  QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), x.data());
  std::vector<UR> b0          = MatrixMultiply<UR, UR>(Ap_, Ai_, Ax_, x);
  std::vector<UR> r           = VectorSubtract<UR>(b, b0);
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
    r           = VectorSubtract<UR>(b, b0);
    d_norm_prev = d_norm;
  }
  return x;
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

  QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), r.data());
  UR rho = Dnrm2<UR>(r);
  if (rho < tol_) {
    return res;
  }

  std::vector<std::vector<UW>> v;
  v.reserve(gmres_iter_ + 1);
  v.emplace_back(n_, 0);
  std::transform(r.cbegin(), r.cend(), v[0].begin(),
                 [rho](UR ri) { return static_cast<UW>(ri / rho); });

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

    // ********** evil tmp
    std::vector<UR> tmp(v[k + 1].size());
    std::transform(v[k + 1].cbegin(), v[k + 1].cend(), tmp.begin(),
                   [](UW vi) { return static_cast<UR>(vi); });
    // Apply preconditioner
    QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(),
                tmp.data());
    std::transform(tmp.cbegin(), tmp.cend(), v[k + 1].begin(),
                   [](UR tmpi) { return static_cast<UW>(tmpi); });
    // ********** end of evil tmp

    UW normav = Dnrm2<UW>(v[k + 1]);

    // Modified Gram-Schmidt (MGS)
    for (std::size_t j = 0; j <= k; j++) {
      h[j][k]  = VectorDot<UW>(v[k + 1], v[j]);
      v[k + 1] = VectorSubtract<UW>(v[k + 1], VectorScale<UW>(v[j], h[j][k]));
    }
    UW normav2 = h[k + 1][k] = Dnrm2<UW>(v[k + 1]);

    // Brown/Hindmarsh condition for reorthogonalization
    if (normav + static_cast<UW>(0.001) * normav2 == normav) {
      for (std::size_t j = 0; j <= k; j++) {
        UW hr = VectorDot<UW>(v[k + 1], v[j]);
        h[j][k] += hr;
        v[k + 1] = VectorSubtract<UW>(v[k + 1], VectorScale<UW>(v[j], hr));
      }
      h[k + 1][k] = Dnrm2<UW>(v[k + 1]);
    }

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
      UW w2    = s[k] * g[k] + s[k] * g[k + 1];
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

// End of GMRES_IR_HPP
#endif
