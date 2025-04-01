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

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
class GmresLDLIR {
 private:
  std::size_t              n_;
  std::vector<std::size_t> Ap_;
  std::vector<std::size_t> Ai_;
  std::vector<UW>          Ax_;

  std::vector<std::size_t> Lp_;
  std::vector<std::size_t> Li_;
  std::vector<UF>          Lx_;
  std::vector<UF>          D_;
  std::vector<UF>          Dinv_;

  std::size_t ir_iter_    = 10;
  std::size_t gmres_iter_ = 10;
  UR          tol_        = 1e-10;

  std::vector<UR> PrecondGmres(const std::vector<UW> &x0,
                               const std::vector<UR> &b);

 public:
  GmresLDLIR() = default;
  void            Compute(std::vector<std::size_t> Ap,
                          std::vector<std::size_t> Ai,
                          std::vector<UW>          Ax);
  std::vector<UW> Solve(const std::vector<UW> &b);

  void SetMaxIRIterations(std::size_t n) { ir_iter_ = n; }
  void SetMaxGmresIterations(std::size_t n) { gmres_iter_ = n; }
  void SetTolerance(UR tol) { tol_ = tol; }
};

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
void GmresLDLIR<UF, UW, UR>::Compute(std::vector<std::size_t> Ap,
                                     std::vector<std::size_t> Ai,
                                     std::vector<UW>          Ax) {
  n_  = Ap.size() - 1;
  Ap_ = std::move(Ap);
  Ai_ = std::move(Ai);
  Ax_ = std::move(Ax);

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
  std::vector<UR> b0 = MatrixMultiply<UR>(Ap_, Ai_, Ax_, x);
  std::vector<UR> r  = VectorSubtract<UR>(b, b0);
  std::cout << Dnrm2<UR>(r) << std::endl;
  for (std::size_t _ = 0; _ < ir_iter_ && Dnrm2<UR>(r) > tol_; _++) {
    UR r_infnorm       = InfNrm(r);
    r                  = VectorScale<UR>(r, static_cast<UR>(1) / r_infnorm);
    std::vector<UR> d  = PrecondGmres(std::vector<UW>(), r);
    d                  = VectorScale<UR>(d, r_infnorm);
    std::vector<UR> xd = VectorAdd<UR>(x, d);
    std::transform(xd.cbegin(), xd.cend(), x.begin(),
                   [](UR x) { return static_cast<UW>(x); });
    b0 = MatrixMultiply<UR>(Ap_, Ai_, Ax_, x);
    r  = VectorSubtract<UR>(b, b0);
    std::cout << Dnrm2<UR>(r) << std::endl;
  }
  return x;
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UR> GmresLDLIR<UF, UW, UR>::PrecondGmres(const std::vector<UW> &x0,
                                                     const std::vector<UR> &b) {
  if (b.size() != n_) {
    // TODO set solver info
    return std::vector<UR>();
  }

  std::vector<UR> res(n_, 0);

  std::vector<UR> r(n_);
  if (x0.empty()) {
    r = b;
  } else {
    std::vector<UR> b0 = MatrixMultiply<UR>(Ap_, Ai_, Ax_, x0);
    r                  = VectorSubtract<UR>(b, b0);
    std::transform(x0.cbegin(), x0.cend(), res.begin(),
                   [](UW x) { return static_cast<UR>(x); });
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
    v[k + 1] = MatrixMultiply<UW>(Ap_, Ai_, Ax_, v[k]);
    // Apply preconditioner
    QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(),
                v[k + 1].data());
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
      h[k + 1][k] = 0;

      UW w1    = c[k] * g[k] - s[k] * g[k + 1];
      UW w2    = s[k] * g[k] + s[k] * g[k + 1];
      g[k]     = w1;
      g[k + 1] = w2;
    }
    rho = std::abs(g[k + 1]);
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
      res[i] += static_cast<UR>(v[j][i]) * static_cast<UR>(y[j]);
    }
  }
  return res;
}

// End of GMRES_IR_HPP
#endif
