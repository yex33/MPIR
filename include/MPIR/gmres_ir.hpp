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

#ifdef QD
#include <qd/dd_real.h>
#include <qd/qd_real.h>
#pragma omp declare reduction(+ : dd_real : omp_out += omp_in) \
    initializer(omp_priv = dd_real(0.0))
#pragma omp declare reduction(+ : qd_real : omp_out += omp_in) \
    initializer(omp_priv = qd_real(0.0))
#endif

/**
 * @brief GMRES with Iterative Refinement and ILU/IC Preconditioning in
 * mixed-precisions.
 *
 * This class implements a GMRES solver with iterative refinement (IR),
 * using an incomplete LU/LDL factorization as a preconditioner.
 *
 * Matrix A is assumed square (n × n) and symmetric. Only its upper triangular
 * part is stored in **CSC format**. Preconditioner factors L and U are stored
 * in mixed CSR/CSC formats for efficient triangular solves:
 *
 *   A ≈ L * U
 *
 * Storage layout:
 *
 *   - Input matrix A: CSC (Ap_, Ai_, Ax_)
 *   - Factor U      : CSC (Up_, Ui_, Ux_)
 *   - Factor L      : CSR (Lp_, Li_, Lx_)
 *
 * CSC format (Compressed Sparse Column):
 *
 *   Ap[j]..Ap[j+1]-1 → nonzeros of column j
 *   Ai[k]            → row index of Ax[k]
 *   Ax[k]            → value
 *
 * CSR format (Compressed Sparse Row):
 *
 *   Lp[i]..Lp[i+1]-1 → nonzeros of row i
 *   Li[k]            → column index of Lx[k]
 *   Lx[k]            → value
 *
 * Diagonal entries of U are stored separately in UDinv_ as their inverses
 * for fast backward substitution.
 *
 * @tparam UF Factorization precision type (low precision, e.g. float, double)
 * @tparam UW Working precision type (e.g. double, quad)
 * @tparam UR Residual precision type (e.g. quad)
 */
template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
class GmresLDLIR {
  std::size_t n_ = 0;  ///< Dimension of the system matrix A (n × n).

  // ----------------- Input matrix A (CSC format) -----------------
  std::vector<std::size_t> Ap_;  ///< Column pointers of A (CSC), size n_+1.
  std::vector<std::size_t> Ai_;  ///< Row indices of A’s nonzeros (CSC).
  std::vector<UW>          Ax_;  ///< Nonzero values of A (CSC).
  std::vector<UW>          AD_;  ///< Cached diagonal entries of A.

  // ----------------- Factor U (upper-triangular, CSC) -------------
  std::vector<std::size_t> Up_;     ///< Column pointers of U (CSC), size n_+1.
  std::vector<std::size_t> Ui_;     ///< Row indices of U’s nonzeros (CSC).
  std::vector<UF>          Ux_;     ///< Nonzero values of U (CSC).
  std::vector<UF>          UDinv_;  ///< Inverses of diagonal entries of U.

  // ----------------- Factor L (lower-triangular, CSR) -------------
  std::vector<std::size_t> Lp_;  ///< Row pointers of L (CSR), size n_+1.
  std::vector<std::size_t> Li_;  ///< Column indices of L’s nonzeros (CSR).
  std::vector<UF>          Lx_;  ///< Nonzero values of L (CSR).

  // ----------------- Solver configuration ------------------------
  std::size_t ir_iter_ =
      10;  ///< Max outer iterations for iterative refinement.
  std::size_t gmres_iter_ =
      10;           ///< Max inner GMRES iterations per refinement step.
  UR tol_ = 1e-10;  ///< Convergence tolerance on residual norm.

  const UW STAGNATE_THRESHOLD = 1;

  /**
   * @brief Solves the linear system A x = b using left-preconditioned GMRES.
   *
   * This function implements **one iteration** of the restarted GMRES
   * (Generalized Minimal Residual) algorithm with an LU/IC preconditioner
   * applied at each iteration.
   *
   * The algorithm:
   *  - Initializes the Krylov subspace with the preconditioned right-hand side
   * b.
   *  - Iteratively builds an orthonormal Krylov basis using Modified
   * Gram-Schmidt.
   *  - Applies Givens rotations to maintain an upper Hessenberg least-squares
   * problem.
   *  - Uses the preconditioner (TriangularSolve) on every matrix-vector
   * product.
   *  - Stops when the residual norm drops below the tolerance @c tol_ or when
   *    @c gmres_iter_ iterations are reached.
   *
   * @param b Right-hand side vector of length n_.
   * @return Approximate solution vector x of length n_.
   *
   * @note
   * - The Krylov subspace size is controlled by @c gmres_iter_.
   * - Uses Modified Gram-Schmidt with optional reorthogonalization
   *   (Brown/Hindmarsh criterion).
   * - The residual norm can fluctuate between iterations due to finite
   * precision, but overall convergence is expected if the preconditioner is
   * effective.
   *
   * @complexity
   * - Each GMRES iteration costs:
   *   - One sparse matrix-vector product (O(nnz(A))),
   *   - One preconditioner application via triangular solves (O(nnz(L) +
   * nnz(U))),
   *   - Orthogonalization against k previous vectors (O(n·k)).
   * - The total per restart is approximately O(k·nnz(A) + n·k²),
   *   where k = @c gmres_iter_ and n = matrix size.
   *
   * @see TriangularSolve, MatrixMultiply
   */
  std::vector<UW> PrecondGmres(const std::vector<UW> &b);

  /**
   * @brief Solves a linear system using the ILU/IC preconditioner.
   *
   * Performs a two-step triangular solve:
   *   1. Forward substitution: solve L * y = b, where L is lower-triangular
   * (CSR).
   *   2. Backward substitution: solve U * x = y, where U is upper-triangular
   * (CSC).
   *
   * Diagonal entries of U are pre-inverted and stored in UDinv_ for efficiency.
   *
   * @tparam T Numeric type of the solve. Deduced from the value type of b.
   * @param b Right-hand side vector of size n_.
   * @return Solution vector x of size n_.
   *
   * @note Assumes that L has implicit unit diagonal and that U’s diagonal
   *       entries are nonzero (UDinv_ is valid).
   */
  template <typename T>
  std::vector<T> TriangularSolve(const std::vector<T> &b);

  /**
   * @brief Computes a truncated dot product between row of L and column of U.
   *
   * Given row index @p row (from L, CSR) and column index @p col (from U, CSC),
   * computes the inner product:
   *
   *    sum_{k < cut} L(row, k) * U(k, col)
   *
   * where @p cut is an index limit that restricts accumulation to the
   * first part of the factorization.
   *
   * @param row Row index into L.
   * @param col Column index into U.
   * @param cut Upper limit on the summation index (exclusive).
   * @return Dot product value in precision type UF.
   *
   * @note This is a sparse variant of a Schur complement update, used in
   *       nonlinear ILU/IC sweeps to update entries.
   */
  UF SparseLDotU(std::size_t row, std::size_t col, std::size_t cut);

  /**
   * @brief Computes the nonlinear residual norm of the current factorization.
   *
   * Evaluates the Frobenius-like residual of A ≈ L * U by measuring
   * entrywise differences:
   *
   *    || A - L * U ||₁ (restricted to observed entries)
   *
   * The residual is accumulated for both upper and lower parts of A:
   *   - For U (CSC): compares each A(row, col) with dot products of L and U.
   *   - For L (CSR): compares each A(col, row) with dot products of L and U.
   *
   * @return Scalar residual norm in precision type UW.
   *
   * @note Parallelized with OpenMP reduction.
   */
  UW NLResNorm();

  /**
   * @brief Retrieves the value of the original sparse matrix A at (row, col).
   *
   * Performs a lookup in the CSC representation of A:
   *   - Scans the range Ap_[col] .. Ap_[col+1] for an entry with row index @p
   * row.
   *   - Returns the stored value Ax_[idx] if found, otherwise returns 0.
   *
   * @param row Row index in A.
   * @param col Column index in A.
   * @return Value of A(row, col) as stored in Ax_, or 0 if structurally zero.
   *
   * @note
   * - Complexity is O(nnz_in_column), since it performs a linear search
   *   over all nonzeros in column @p col.
   * - Return type matches the scalar type stored in Ax_ (i.e.,
   * decltype(Ax_)::value_type).
   */
  auto ValueAt(std::size_t row, std::size_t col) -> decltype(Ax_)::value_type;

 public:
  GmresLDLIR() = default;
  /**
   * @brief Computes and stores the ILU/IC factorization of the given sparse
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
          UDinv_[col] = static_cast<UF>(1) / Ux_[idx1];  // Memorize the inverse
        } else {
          // Off diagonal
          Ux_[idx1] = static_cast<UF>(Ax_[idx2]);
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

  std::println("Nonlinear residual norm @ 0 sweep is {:g}", NLResNorm());

  constexpr std::size_t NUM_SWEEPS = 10;
  for (std::size_t sweep = 0; sweep < NUM_SWEEPS; sweep++) {
    std::vector<UF> Ux_new(Ux_.size()), Lx_new(Lx_.size());
    std::vector<UF> UDinv_new(n_);

#pragma omp parallel for schedule(dynamic, 4096)
    for (std::size_t U_col = 0; U_col < n_; U_col++) {
      for (std::size_t idx = Up_[U_col]; idx < Up_[U_col + 1]; idx++) {
        const std::size_t U_row = Ui_[idx];
        const UF ax = static_cast<UF>(ValueAt(U_row, U_col));
        // Find A(row, col)
        {
          const UF sum = SparseLDotU(U_row, U_col, U_row);
          // Fill non-linear solution
          Ux_new[idx] = ax - sum;
          if (U_row == U_col) {
            UDinv_new[U_row] = static_cast<UF>(1) / Ux_new[idx];
            Lx_new[idx] = 1;
          }
        }

        if (U_row != U_col) {
          // Translate indices from U to L
          const std::size_t L_row = U_col, L_col = U_row;

          const UF sum = SparseLDotU(L_row, L_col, L_col);
          // Fill non-linear solution
          Lx_new[idx] = (ax - sum) * UDinv_[L_col];
        }
      }
    }

    Ux_    = Ux_new;
    Lx_    = Lx_new;
    UDinv_ = UDinv_new;

    std::println("Nonlinear residual norm @ {} sweep is {:g}", sweep + 1,
                 NLResNorm());
    std::println("norm(UDinv) = {:g}", InfNrm(UDinv_));
  }  // End of sweep
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::Solve(const std::vector<UW> &b) {
  using std::abs;

  if (ir_iter_ == 0) {
  }
  std::vector<UW> b_scaled = VectorMultiply<UW>(b, AD_);

  std::vector<UW> x  = PrecondGmres(b_scaled);
  std::vector<UR> b0 = MatrixMultiply<UR>(Ap_, Ai_, Ax_, x);
  std::vector<UR> r  = VectorSubtract<UR>(b_scaled, b0);
  std::println("first residual {:g}", InfNrm(r));
  UW d_norm_prev = std::numeric_limits<UW>::max();
  UR r_norm_prev = std::numeric_limits<UR>::max();
  for (std::size_t _ = 0; _ < ir_iter_; _++) {
    UR              scale = InfNrm(r);
    std::vector<UW> b     = VectorScale<UW, UR>(r, static_cast<UR>(1) / scale);
    std::vector<UW> d     = PrecondGmres(b);
    d                     = VectorScale<UW>(d, scale);
    UW d_norm             = InfNrm(d);
    UW x_norm             = InfNrm(x);
    // std::cout << "residual " << static_cast<double>(d_norm) << std::endl;
    // || d_norm >= 0.5 * d_norm_prev
    x         = VectorAdd<UW>(x, d);
    b0        = MatrixMultiply<UR>(Ap_, Ai_, Ax_, x);
    r         = VectorSubtract<UR>(b_scaled, b0);
    UR r_norm = InfNrm(r);
    std::println("residual {:g}", InfNrm(r));
    if (d_norm <= x_norm * (10 * std::numeric_limits<UW>::epsilon()) ||
        abs(r_norm - r_norm_prev) < tol_) {
      break;
    }
    r_norm_prev = r_norm;
    d_norm_prev = d_norm;
  }
  return VectorMultiply<UW>(x, AD_);
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
std::vector<UW> GmresLDLIR<UF, UW, UR>::PrecondGmres(const std::vector<UW> &b) {
  if (b.size() != n_) {
    // TODO set solver info
    return {};
  }

  std::vector<UW> res(n_, 0);

  UW rho = Dnrm2<UW>(b);
  if (rho < tol_) {
    return res;
  }

  std::vector<std::vector<UW>> v;
  v.reserve(gmres_iter_ + 1);
  v.push_back(b);
  v[0] = TriangularSolve(v[0]);
  rho  = Dnrm2<UW>(v[0]);
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
    if (normav + static_cast<UW>(0.001) * normav2 == normav) {
      std::println("reorth triggered at iteration {}", k + 1);
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
template <typename T>
std::vector<T> GmresLDLIR<UF, UW, UR>::TriangularSolve(
    const std::vector<T> &b) {
  using std::isnan;

  std::vector<T> y(n_);

  // Forward solve: L y = b (lower-triangular system)
  for (std::size_t row = 0; row < n_; row++) {
    T sum = static_cast<T>(0);
    for (std::size_t idx = Lp_[row]; idx < Lp_[row + 1]; idx++) {
      if (const std::size_t col = Li_[idx]; col < row) {
        sum += Lx_[idx] * y[col];
      }
    }
    y[row] = b[row] - sum;
  }

  std::vector<T> x = y;
  // Backward solve: U x = y (upper-triangular system)
  for (std::size_t col = n_; col-- > 0;) {
    x[col] *= UDinv_[col];
    if (isnan(x[col])) {
      std::println("{}", col);
      exit(1);
    }
    for (std::size_t idx = Up_[col]; idx < Up_[col + 1]; idx++) {
      if (const size_t row = Ui_[idx]; row < col) {
        x[row] -= Ux_[idx] * x[col];
      }
    }
  }
  return x;
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
UW GmresLDLIR<UF, UW, UR>::NLResNorm() {
  UW nonlinear_residual_norm = 0.0;
#pragma omp parallel for reduction(+ : nonlinear_residual_norm) \
    schedule(dynamic, 8)
  for (std::size_t col = 0; col < n_; col++) {
    for (std::size_t idx = Up_[col]; idx < Up_[col + 1]; idx++) {
      const std::size_t row = Ui_[idx];
      // Find A(row, col)
      const UF ax  = static_cast<UF>(ValueAt(row, col));
      const UF sum = SparseLDotU(row, col, row + 1);
      nonlinear_residual_norm += abs(ax - sum);
    }
  }
#pragma omp parallel for reduction(+ : nonlinear_residual_norm) \
    schedule(dynamic, 8)
  for (std::size_t row = 0; row < n_; row++) {
    for (std::size_t idx = Lp_[row]; idx < Lp_[row + 1]; idx++) {
      const std::size_t col = Li_[idx];
      if (col >= row) continue;
      // Find A(col, row)
      const UF ax  = static_cast<UF>(ValueAt(col, row));
      const UF sum = SparseLDotU(row, col, col + 1);
      nonlinear_residual_norm += abs(ax - sum);
    }
  }
  return nonlinear_residual_norm;
}

template <typename UF, typename UW, typename UR>
  requires Refinable<UF, UW, UR>
auto GmresLDLIR<UF, UW, UR>::ValueAt(const std::size_t row,
                                     const std::size_t col)
    -> decltype(Ax_)::value_type {
  for (std::size_t idx = Ap_[col]; idx < Ap_[col + 1]; idx++) {
    if (Ai_[idx] == row) {
      return Ax_[idx];
    }
  }
  return static_cast<decltype(Ax_)::value_type>(0.0);
}

// End of GMRES_IR_HPP
#endif
