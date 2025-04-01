#pragma once

#ifndef GMRES_IR_HPP
#define GMRES_IR_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

#include "qdldl.hpp"

template <typename T, typename... Ts>
constexpr bool is_all_floating_point_v =
    std::floating_point<T> && is_all_floating_point_v<Ts...>;

template <typename T>
constexpr bool is_all_floating_point_v<T> = std::floating_point<T>;

template <typename... Ts>
concept FloatingPoint = is_all_floating_point_v<Ts...>;

template <typename T, typename... Ts>
constexpr bool is_partial_ordered_v = ((std::numeric_limits<T>().epsilon() >=
                                        std::numeric_limits<Ts>().epsilon()) &&
                                       ...) &&
                                      is_partial_ordered_v<Ts...>;

template <typename T>
constexpr bool is_partial_ordered_v<T> = true;

template <typename... Ts>
concept PartialOrdered = is_partial_ordered_v<Ts...>;

template <typename... Ts>
concept Refinable = FloatingPoint<Ts...> && PartialOrdered<Ts...>;

template <typename T, typename Ta, typename Tx>
  requires Refinable<Ta, Tx, T>
std::vector<T> MatrixMultiply(const std::vector<std::size_t> &Ap,
                              const std::vector<std::size_t> &Ai,
                              const std::vector<Ta>          &Ax,
                              const std::vector<Tx>          &x);

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
std::vector<T> VectorSubtract(const std::vector<Ta> &a,
                              const std::vector<Tb> &b);

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
std::vector<T> VectorMultiply(const std::vector<Ta> &a,
                              const std::vector<Tb> &b);

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
std::vector<T> VectorScale(const std::vector<Ta> &a, Tb b);

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
T VectorDot(const std::vector<Ta> &a, const std::vector<Tb> &b);

template <typename T, typename Tx>
  requires Refinable<Tx, T>
T Dnrm2(const std::vector<Tx> &x);

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

  std::vector<UW> PrecondGmres(const std::vector<UW> &x0,
                               const std::vector<UW> &b);

 public:
  GmresLDLIR() = default;
  void Compute(std::vector<std::size_t> Ap,
               std::vector<std::size_t> Ai,
               std::vector<UW>          Ax);
  void Solve(const std::vector<UW> &b);

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
void GmresLDLIR<UF, UW, UR>::Solve(const std::vector<UW> &b) {
  PrecondGmres(std::vector<UW>(), b);
  if (ir_iter_ == 0) {
  }
  for (std::size_t _ = 0; _ < ir_iter_; _++) {
  }
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::PrecondGmres(const std::vector<UW> &x0,
                                                     const std::vector<UW> &b) {
  if (b.size() != n_) {
    // TODO set solver info
    return std::vector<UW>();
  }
  std::vector<UR> r(n_);
  if (x0.size() == 0) {
    std::transform(b.cbegin(), b.cend(), r.begin(),
                   [](UW x) { return static_cast<UR>(x); });
  } else {
    std::vector<UR> b0 = MatrixMultiply<UR>(Ap_, Ai_, Ax_, x0);
    r                  = VectorSubtract<UR>(b, b0);
  }

  QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), r.data());
  UR rho = Dnrm2<UR>(r);
  if (rho < tol_) {
    return x0;
  }

  std::vector<std::vector<UW>> h(gmres_iter_ + 1,
                                 std::vector<UW>(gmres_iter_, 0));

  std::vector<std::vector<UW>> v(gmres_iter_ + 1, std::vector<UW>(n_, 0));
  std::transform(r.cbegin(), r.cend(), v[0].begin(),
                 [rho](UR ri) { return static_cast<UW>(ri / rho); });

  std::vector<UW> g(gmres_iter_ + 1, 0);
  std::vector<UW> c(gmres_iter_, 0);
  std::vector<UW> s(gmres_iter_, 0);
  g[0] = static_cast<UW>(rho);

  for (std::size_t k = 0; k < gmres_iter_ && rho > tol_; k++) {
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
    for (std::size_t j = 0; k > 0 && j <= k; j++) {
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
    std::cout << rho << std::endl;
  }

  return std::vector<UW>();
}

template <typename T, typename Ta, typename Tx>
  requires Refinable<Ta, Tx, T>
std::vector<T> MatrixMultiply(const std::vector<std::size_t> &Ap,
                              const std::vector<std::size_t> &Ai,
                              const std::vector<Ta>          &Ax,
                              const std::vector<Tx>          &x) {
  std::vector<T> y(x.size());
  const size_t   n = Ap.size() - 1;
  for (std::size_t col = 0; col < n; col++) {
    for (std::size_t idx = Ap[col]; idx < Ap[col + 1]; idx++) {
      const size_t row = Ai[idx];
      y[col] += static_cast<T>(Ax[idx]) * static_cast<T>(x[row]);
      if (row < col) {
        y[row] += static_cast<T>(Ax[idx]) * static_cast<T>(x[col]);
      }
    }
  }
  return y;
}

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
std::vector<T> VectorSubtract(const std::vector<Ta> &a,
                              const std::vector<Tb> &b) {
  std::vector<T> res(a.size());
  std::transform(
      a.cbegin(), a.cend(), b.cbegin(), res.begin(),
      [](Ta ai, Tb bi) { return static_cast<T>(ai) - static_cast<T>(bi); });
  return res;
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[MPIR] VectorSubtract") {
  std::vector<double> v1      = {5.5, 6.6, 7.7};
  std::vector<double> v2      = {1.1, 2.2, 3.3};
  auto                result1 = VectorSubtract<double>(v1, v2);
  CHECK(std::equal(result1.begin(), result1.end(),
                   std::vector<double>{4.4, 4.4, 4.4}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));

  std::vector<double> v3      = {10.0, 20.0, 30.0};
  std::vector<double> v4      = {5.0, 10.0, 15.0};
  auto                result2 = VectorSubtract<double>(v3, v4);
  CHECK(std::equal(result2.begin(), result2.end(),
                   std::vector<double>{5.0, 10.0, 15.0}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));
}
#endif

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
std::vector<T> VectorMultiply(const std::vector<Ta> &a,
                              const std::vector<Tb> &b) {
  std::vector<T> res(a.size());
  std::transform(
      a.cbegin(), a.cend(), b.cbegin(), res.begin(),
      [](Ta ai, Tb bi) { return static_cast<T>(ai) * static_cast<T>(bi); });
  return res;
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[MPIR] VectorMultiply") {
  std::vector<double> v1      = {1.1, 2.2, 3.3};
  std::vector<double> v2      = {4.4, 5.5, 6.6};
  auto                result1 = VectorMultiply<double>(v1, v2);
  CHECK(std::equal(result1.begin(), result1.end(),
                   std::vector<double>{4.84, 12.1, 21.78}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));

  std::vector<double> v3      = {2.0, 4.0, 6.0};
  std::vector<double> v4      = {1.5, 2.5, 3.5};
  auto                result2 = VectorMultiply<double>(v3, v4);
  CHECK(std::equal(result2.begin(), result2.end(),
                   std::vector<double>{3.0, 10.0, 21.0}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));
}
#endif

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
std::vector<T> VectorScale(const std::vector<Ta> &a, Tb b) {
  std::vector<T> res(a.size());
  std::transform(a.cbegin(), a.cend(), res.begin(),
                 [b](Ta ai) { return static_cast<T>(ai) * static_cast<T>(b); });
  return res;
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[MPIR] VectorScale") {
  std::vector<double> v1           = {1.1, 2.2, 3.3};
  double              scale_factor = 2.0;
  auto                result1      = VectorScale<double>(v1, scale_factor);
  CHECK(std::equal(result1.begin(), result1.end(),
                   std::vector<double>{2.2, 4.4, 6.6}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));

  std::vector<double> v2            = {3.0, 6.0, 9.0};
  double              scale_factor2 = 0.5;
  auto                result2       = VectorScale<double>(v2, scale_factor2);
  CHECK(std::equal(result2.begin(), result2.end(),
                   std::vector<double>{1.5, 3.0, 4.5}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));
}
#endif

template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
T VectorDot(const std::vector<Ta> &a, const std::vector<Tb> &b) {
  return std::transform_reduce(a.cbegin(), a.cend(), b.cbegin(),
                               static_cast<T>(0));
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[MPIR] VectorDot") {
  std::vector<double> v1 = {1.1, 2.2, 3.3};
  std::vector<double> v2 = {4.4, 5.5, 6.6};
  CHECK_EQ(VectorDot<double>(v1, v2),
           doctest::Approx(38.72));  // 1.1*4.4 + 2.2*5.5 + 3.3*6.6

  std::vector<float> v3 = {1.1f, 2.2f, 3.3f};
  std::vector<float> v4 = {4.4f, 5.5f, 6.6f};
  CHECK_EQ(VectorDot<float>(v3, v4), doctest::Approx(38.72f));

  std::vector<float>  v5 = {1.5f, 2.5f, 3.5f};
  std::vector<double> v6 = {4.5, 5.5, 6.5};
  CHECK_EQ(VectorDot<double>(v5, v6),
           doctest::Approx(43.25));  // 1.5*4.5 + 2.5*5.5 + 3.5*6.5

  std::vector<double> empty1;
  std::vector<double> empty2;
  CHECK_EQ(VectorDot<double>(empty1, empty2),
           0.0);  // Dot product of empty vectors should be 0
}
#endif

template <typename T, typename Tx>
  requires Refinable<Tx, T>
T Dnrm2(const std::vector<Tx> &x) {
  if (x.size() == 0) {
    return static_cast<T>(0);
  } else if (x.size() == 1) {
    return std::abs(static_cast<T>(x[0]));
  }
  T scale = static_cast<T>(0);
  T ssq   = static_cast<T>(1);
  for (std::size_t i = x.size(); i-- > 0;) {
    if (static_cast<T>(x[i]) != static_cast<T>(0)) {
      T x_abs = std::abs(static_cast<T>(x[i]));
      if (scale < x_abs) {
        ssq   = static_cast<T>(1) + ssq * ((scale / x_abs) * (scale / x_abs));
        scale = x_abs;
      } else {
        ssq = ssq + ((x_abs / scale) * (x_abs / scale));
      }
    }
  }
  return scale * std::sqrt(ssq);
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[MPIR] Dnrm2") {
  std::vector<double> v1 = {3.0, 4.0};
  CHECK_EQ(Dnrm2<double>(v1), doctest::Approx(5.0));  // sqrt(3^2 + 4^2) = 5

  std::vector<double> v2 = {1.0, 2.0, 2.0};
  CHECK_EQ(Dnrm2<double>(v2),
           doctest::Approx(3.0));  // sqrt(1^2 + 2^2 + 2^2) = 3

  std::vector<double> v3 = {0.0, 0.0, 0.0};
  CHECK_EQ(Dnrm2<double>(v3), 0.0);  // Norm of zero vector is 0

  std::vector<double> v4 = {5.5};
  CHECK_EQ(Dnrm2<double>(v4), doctest::Approx(5.5));  // Single element case

  std::vector<double> v5 = {3.14, 2.71, 1.41};
  CHECK_EQ(Dnrm2<double>(v5),
           doctest::Approx(std::sqrt(3.14 * 3.14 + 2.71 * 2.71 + 1.41 * 1.41)));

  std::vector<double> v6 = {0.577, 0.577, 0.577};
  CHECK_EQ(Dnrm2<double>(v6),
           doctest::Approx(
               std::sqrt(0.577 * 0.577 + 0.577 * 0.577 + 0.577 * 0.577)));

  std::vector<double> empty;
  CHECK_EQ(Dnrm2<double>(empty), 0.0);  // Norm of empty vector should be 0
}
#endif

// End of GMRES_IR_HPP
#endif
