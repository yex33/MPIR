#ifndef TBLAS_H
#define TBLAS_H
#include <chrono>
#include <qd/dd_real.h>
#include <qd/qd_real.h>

// Simple BLAS-like functions

// This is needed to be consistent with the to_double from the QD package.
inline double to_double(const double d) { return d; }
#define ABS(x) ((x >= 0) ? x : -x)

/**
 * Calculates the infinity norm of a vector.
 *
 * @param N The size of the vector.
 * @param x Pointer to the vector.
 * @return The infinity norm of the vector.
 */
template <typename T>
inline double inf_norm(int N, const T* x)
{
  T max = 0;
  for (int i = 0; i < N; i++)
    {
      if (ABS(x[ i ]) > max) max = ABS(x[ i ]);
    }
  return to_double(max);
}

/**
 * Scales each element of the array `x` by the scalar `s`.
 *
 * @tparam INT The integer type used for indexing the array.
 * @tparam T The type of the elements in the array.
 * @param s The scalar value to scale the elements by.
 * @param n The number of elements in the array.
 * @param x The array to be scaled.
 */
template <typename INT, typename T>
void scale(double s, INT n, T* x)
{
  for (INT i = 0; i < n; i++) x[ i ] *= s;
}

// d <- s
template <typename INT, typename SOURCE, typename DEST>
inline void copy_vector(INT n, const SOURCE* s, DEST* d)
{
  // This "sophistication" is needed because of the conversions in the QD
  // package to_double and to_dd_real.
  // float|double <- float|double|dd_real|qd_real
  if constexpr ((std::is_same_v<DEST, float> || std::is_same_v<DEST, double>))
    {
      for (INT i = 0; i < n; i++) d[ i ] = to_double(s[ i ]);
    }
  // dd_real <- qd_real
  else if constexpr ((
                      std::is_same_v<SOURCE, qd_real>))
    {
      for (INT i = 0; i < n; i++) d[ i ] = to_dd_real(s[ i ]);
    }
  else
    for (INT i = 0; i < n; i++) d[ i ] = s[ i ];
}

/**
 * Adds two vectors element-wise and stores the result in a third vector.
 *
 * @tparam PREC The precision type of the input vectors.
 * @tparam INT The integer type used for indexing.
 * @tparam Ta The type of elements in the first input vector.
 * @tparam Tb The type of elements in the second input vector.
 * @tparam Tc The type of elements in the output vector.
 * @param n The number of elements in the vectors.
 * @param a Pointer to the first input vector.
 * @param b Pointer to the second input vector.
 * @param c Pointer to the output vector.
 */
template <typename PREC, typename INT, typename Ta, typename Tb, typename Tc>
inline void add_vector(INT n, const Ta* a, Tb* b, Tc* c)
{
  for (INT i = 0; i < n; i++)
    {
      PREC tmp;
      if constexpr (std::is_same_v<PREC, float> || std::is_same_v<PREC, double>)
        tmp = to_double(a[ i ]) + to_double(b[ i ]);
      else
        tmp = PREC(a[ i ]) + PREC(b[ i ]);
      if constexpr (std::is_same_v<Tc, float> || std::is_same_v<Tc, double>)
        c[ i ] = to_double(tmp);
      else if constexpr (std::is_same_v<Tc, dd_real> &&
                         std::is_same_v<PREC, qd_real>)
        c[ i ] = to_dd_real(tmp);
      else
        c[ i ] = Tc(tmp);
    }
}

#define SUB(conversion)           \
  for (INT i = 0; i < n; i++)     \
    {                             \
      PREC x = a[ i ];            \
      PREC y = b[ i ];            \
      c[ i ] = conversion(x - y); \
    }

template <typename PREC, typename INT, typename Ta, typename Tb, typename Tc>
inline void sub_vector(INT n, const Ta* a, const Tb* b, Tc* c)
{
  if constexpr (std::is_same_v<Tc, float> || std::is_same_v<Tc, double>)
    {
      SUB(to_double)
    }
  else if constexpr (std::is_same_v<Tc, dd_real> &&
                     std::is_same_v<PREC, qd_real>)
    {
      SUB(to_dd_real);
    }
  else
    SUB(Tc);
}

template <typename INT, typename X, typename XREF>
double rel_error(INT n, X* x, XREF* xref)
{
  XREF* tmp = new XREF[ n ];
  sub_vector<XREF>(n, x, xref, tmp);
  double error = inf_norm(n, tmp) / inf_norm(n, xref);
  delete[] tmp;
  return error;
}

inline auto get_time() { return std::chrono::steady_clock::now(); };

inline double get_time(auto start)
{
  auto end = std::chrono::steady_clock::now();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  return milliseconds / 1000.0;
}

#endif