#pragma once

#ifndef OPS_HPP
#define OPS_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "fp_concepts.hpp"

/**
 * @brief Multiplies a sparse matrix in CSC format by a vector.
 * @tparam T Precision of the matrix-vector product.
 * @param Ap Column pointers of the matrix in CSC format.
 * @param Ai Row indices of the matrix in CSC format.
 * @param Ax Non-zero values of the matrix in CSC format.
 * @param x Vector being multiplied.
 * @return The matrix-vector product.
 */
template <typename T, typename Ta, typename Tx>
  requires Refinable<Ta, Tx, T>
std::vector<T> MatrixMultiply(const std::vector<std::size_t> &Ap,
                              const std::vector<std::size_t> &Ai,
                              const std::vector<Ta>          &Ax,
                              const std::vector<Tx>          &x) {
  std::vector<T> res(x.size());
  const size_t   n = Ap.size() - 1;
  for (std::size_t col = 0; col < n; col++) {
    for (std::size_t idx = Ap[col]; idx < Ap[col + 1]; idx++) {
      const size_t row = Ai[idx];
      res[col] += static_cast<T>(Ax[idx]) * static_cast<T>(x[row]);
      if (row < col) {
        res[row] += static_cast<T>(Ax[idx]) * static_cast<T>(x[col]);
      }
    }
  }
  return res;
}

/**
 * @brief Adds two vectors element-wise.
 * @tparam T Precision of the sum.
 * @param a First input vector.
 * @param b Second input vector.
 * @return The element-wise sum of vectors a and b.
 */
template <typename T, typename Ta, typename Tb>
  requires Refinable<Ta, T> && Refinable<Tb, T>
std::vector<T> VectorAdd(const std::vector<Ta> &a, const std::vector<Tb> &b) {
  std::vector<T> res(a.size());
  std::transform(
      a.cbegin(), a.cend(), b.cbegin(), res.begin(),
      [](Ta ai, Tb bi) { return static_cast<T>(ai) + static_cast<T>(bi); });
  return res;
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[MPIR] VectorAdd") {
  std::vector<double> v1      = {5.5, 6.6, 7.7};
  std::vector<double> v2      = {1.1, 2.2, 3.3};
  auto                result1 = VectorAdd<double>(v1, v2);
  CHECK(std::equal(result1.begin(), result1.end(),
                   std::vector<double>{6.6, 8.8, 11.0}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));

  std::vector<double> v3      = {10.0, 20.0, 30.0};
  std::vector<double> v4      = {5.0, 10.0, 15.0};
  auto                result2 = VectorAdd<double>(v3, v4);
  CHECK(std::equal(result2.begin(), result2.end(),
                   std::vector<double>{15.0, 30.0, 45.0}.begin(),
                   [](double a, double b) { return doctest::Approx(a) == b; }));
}
#endif

/**
 * @brief Subtracts one vector from another element-wise.
 * @tparam T Precision of the difference.
 * @param a First input vector.
 * @param b Second input vector.
 * @return The element-wise difference of vectors a and b.
 */
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

/**
 * @brief Multiplies two vectors element-wise.
 * @tparam T Precision of the product.
 * @param a First input vector.
 * @param b Second input vector.
 * @return The element-wise product of vectors a and b.
 */
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

/**
 * @brief Scales a vector by a scalar value.
 * @tparam T Precision of the vector after scaling.
 * @param a Input vector.
 * @param b Scalar multiplier.
 * @return The vector a scaled by b.
 */
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

/**
 * @brief Computes the dot product of two vectors.
 * @tparam T Precision of the dot product.
 * @param a First input vector.
 * @param b Second input vector.
 * @return The dot product of vectors a and b.
 */
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

/**
 * @brief Computes the Euclidean norm (2-norm) of a vector.
 * @tparam T Precision of the 2-norm.
 * @param x Input vector.
 * @return The 2-norm of vector x.
 */
template <typename T, typename Tx>
  requires Refinable<Tx, T>
T Dnrm2(const std::vector<Tx> &x) {
  using std::abs;
  using std::sqrt;
  if (x.size() == 0) {
    return static_cast<T>(0);
  } else if (x.size() == 1) {
    return abs(static_cast<T>(x[0]));
  }
  T scale = static_cast<T>(0);
  T ssq   = static_cast<T>(1);
  for (std::size_t i = x.size(); i-- > 0;) {
    if (static_cast<T>(x[i]) != static_cast<T>(0)) {
      T x_abs = abs(static_cast<T>(x[i]));
      if (scale < x_abs) {
        ssq   = static_cast<T>(1) + ssq * ((scale / x_abs) * (scale / x_abs));
        scale = x_abs;
      } else {
        ssq = ssq + ((x_abs / scale) * (x_abs / scale));
      }
    }
  }
  return scale * sqrt(ssq);
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

/**
 * @brief Computes the infinity norm (maximum absolute value) of a vector.
 * @tparam T Precision of the infinity norm.
 * @param x Input vector.
 * @return The infinity norm of vector x.
 */
template <typename T>
T InfNrm(const std::vector<T> &x) {
  using std::abs;
  return abs(*std::max_element(x.cbegin(), x.cend(), [](T a, T b) {
    return abs(a) < abs(b);
  }));
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[MPIR] InfNrm") {
  std::vector<double> v1 = {3.0, -4.0, 5.5, -6.1};
  CHECK_EQ(InfNrm<double>(v1),
           doctest::Approx(6.1));  // Max absolute value is 6.1

  std::vector<double> v2 = {-1.1, -2.2, -3.3};
  CHECK_EQ(InfNrm<double>(v2),
           doctest::Approx(3.3));  // Max absolute value is 3.3

  std::vector<double> v3 = {0.0, 0.0, 0.0};
  CHECK_EQ(InfNrm<double>(v3), 0.0);  // Max absolute value is 0

  std::vector<double> v4 = {7.5};
  CHECK_EQ(InfNrm<double>(v4), doctest::Approx(7.5));  // Single element case

  std::vector<double> v5 = {-3.14, 2.71, -1.41, 5.89};
  CHECK_EQ(InfNrm<double>(v5),
           doctest::Approx(5.89));  // Max absolute value is 5.89
}
#endif

// End of OPS_HPP
#endif
