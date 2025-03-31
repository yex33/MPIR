#pragma once

#ifndef GMRES_IR_HPP
#define GMRES_IR_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <ranges>
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
  requires Refinable<Ta, Tb, T>
std::vector<T> VectorSubtract(const std::vector<Ta> &a,
                              const std::vector<Tb> &b);

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

  std::size_t ir_iter_;
  std::size_t gmres_iter_;
  UR          tol_;

  std::vector<UW> PrecondGmres(const std::vector<UW> &x0,
                               const std::vector<UW> &b);

  template <typename T>
  T Dnrm2(const std::vector<T> &x);

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

  std::vector<UF> D(n_);
  std::vector<UF> Dinv(n_);
  bool           *bwork = new bool[n_];
  std::vector<UF> fwork(n_);
  if constexpr (std::same_as<UF, UW>) {
    QDLDL_factor(n_, Ap_.data(), Ai_.data(), Ax_.data(), Lp_.data(), Li_.data(),
                 Lx_.data(), D.data(), Dinv.data(), Lcolnz.data(), etree.data(),
                 bwork, iwork.data(), fwork.data());
  } else {
    std::vector<UF> Ax_UF(Ax_.size());
    std::transform(Ax_.cbegin(), Ax_.cend(), Ax_UF.begin(),
                   [](UW x) { return static_cast<UF>(x); });
    QDLDL_factor(n_, Ap_.data(), Ai_.data(), Ax_UF.data(), Lp_.data(),
                 Li_.data(), Lx_.data(), D.data(), Dinv.data(), Lcolnz.data(),
                 etree.data(), bwork, iwork.data(), fwork.data());
  }
  delete[] bwork;
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
void GmresLDLIR<UF, UW, UR>::Solve(const std::vector<UW> &b) {
  if (ir_iter_ == 0) {
  }
  for (int _ = 0; _ < ir_iter_; _++) {
  }
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::PrecondGmres(const std::vector<UW> &x0,
                                                     const std::vector<UW> &b) {
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
template <typename T>
T GmresLDLIR<UF, UW, UR>::Dnrm2(const std::vector<T> &x) {
  if (x.size() == 0) {
    return static_cast<T>(0);
  } else if (x.size() == 1) {
    return std::abs(x[0]);
  }
  T scale = static_cast<T>(0);
  T ssq   = static_cast<T>(1);
  for (std::size_t i = x.size(); i-- > 0;) {
    if (x[i] != static_cast<T>(0)) {
      T x_abs = std::abs(x[i]);
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

// End of GMRES_IR_HPP
#endif
