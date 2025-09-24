#ifndef LUIR_H
#define LUIR_H

#include <cassert>
#include <chrono>
#include <limits>
#include <typeinfo>

#include "../MPIR/qdldl.hpp"
#include "residual.hpp"
#include "stats.hpp"
#include "tblas.hpp"

    /**
     * Performs the LUIR  algorithm to solve a linear system of equations.
     *
     * @tparam FACTPREC   The precision type for the factorization.
     * @tparam RESPREC    The precision type for the residual calculation.
     * @tparam WORKPREC   The precision type for the working memory.
     * @tparam XPREC      The precision type for the solution vector.
     * @tparam INT        The integer type for indexing.
     * @param n           The size of the linear system.
     * @param Ap          The column pointers of the sparse matrix A.
     * @param Ai          The row indices of the sparse matrix A.
     * @param Ax          The values of the sparse matrix A.
     * @param b           The right-hand side vector.
     * @param x           The solution vector.
     * @param stats       Pointer to the statistics object.
     * @param maxiter     The maximum number of iterations for the iterative
     * refinement (default is 10).
     */
    /*luir*/
    template <typename FACTPREC,
              typename RESPREC,
              typename WORKPREC,
              typename XPREC,
              typename INT>
    void LUIR_QDLDL(INT             n,
                    const INT*      Ap,
                    const INT*      Ai,
                    const WORKPREC* Ax,
                    const WORKPREC* b,
                    XPREC*          x,
                    Stats*          stats,
                    int             maxiter = 10)
/*luir*/
{
  // auto time         = get_time();

  if constexpr (std::is_same_v<FACTPREC, _Float16>)
    stats->fact_prec_ = "Float16";
  else
    stats->fact_prec_ = typeid(FACTPREC).name();

  stats->work_prec_ = typeid(WORKPREC).name();
  stats->res_prec_  = typeid(RESPREC).name();
  // data for L and D factors
  INT Ln = n;

  /*--------------------------------
   * pre-factorisation memory allocations
   *---------------------------------*/
  // These can happen *before* the etree is calculated
  // since the sizes are not sparsity pattern specific

  // For the elimination tree

  // Working memory.  Note that both the etree and factor
  // calls requires a working vector of int, with
  // the factor function requiring 3*N elements and the
  // etree only N elements.   Just allocate the larger
  // amount here and use it in both places

  /*--------------------------------
   * elimination tree calculation
   *---------------------------------*/
  // double t0=omp_get_wtime();

  auto time_sym = get_time();
  INT* iwork    = new INT[ 3 * n ];
  INT* Lnz      = new INT[ n ];
  INT* etree    = new INT[ n ];
  INT  sumLnz   = QDLDL_etree(n, Ap, Ai, iwork, Lnz, etree);

  stats->time_sym_ = get_time(time_sym);

  if (sumLnz == -1)
    {
      fprintf(stderr, "!!! The input matrix is not in upper triangular form\n");
      exit(-1);
    }
  if (sumLnz == -2)
    {
      fprintf(stderr,
              "!!! total nonzeros in L below diagonal overflows in INT\n");
      exit(-1);
    }
  stats->Lnz_ = sumLnz;

  auto time_nfact = get_time();
  /*
   * LDL factorisation
   *---------------------------------*/
  // First allocate memory for Li and Lx

  INT*      Li = new INT[ sumLnz ];
  FACTPREC* Lx = new FACTPREC[ sumLnz ];
  // For the L factors.   Li and Lx are sparsity dependent
  // so must be done after the etree is constructed
  INT*      Lp    = new INT[ n + 1 ];
  FACTPREC* D     = new FACTPREC[ n ];
  FACTPREC* Dinv  = new FACTPREC[ n ];
  bool*     bwork = new bool[ n ];
  FACTPREC* fwork = new FACTPREC[ n ];

  INT nnz   = Ap[ n ] - Ap[ 0 ];  // number of nonzeros
  stats->n_ = n;

  stats->nnz_ = nnz;
  if constexpr (std::is_same_v<FACTPREC, WORKPREC>)
    {
      // factorization and working precision are the same.
      QDLDL_factor(
          n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
    }
  else
    {
      // factorization and working precision are different.
      // Create the A values in the factorization precision.
      FACTPREC* Afact_prec = new FACTPREC[ nnz ];
      copy_vector(nnz, Ax, Afact_prec);

      QDLDL_factor(n,
                   Ap,
                   Ai,
                   Afact_prec,
                   Lp,
                   Li,
                   Lx,
                   D,
                   Dinv,
                   Lnz,
                   etree,
                   bwork,
                   iwork,
                   fwork);
      delete[] Afact_prec;
    }
  delete[] iwork;
  delete[] Lnz;
  delete[] etree;
  delete[] fwork;
  delete[] bwork;
  stats->time_num_fact_ = get_time(time_nfact);

#define ALLOC(T, var, N) T* var = new T[ N ];
#define DEALLOC(v) delete[] v;

  stats->num_refinement_steps_ = 0;
  ALLOC(FACTPREC, residual_factprec, n);
  if (maxiter == 0)
    {
      auto time_solve = get_time();
      copy_vector(n, b, x);
      QDLDL_solve(Ln, Lp, Li, Lx, Dinv, x);
      ResidualCSCSymmetric<RESPREC, 0>(n, Ax, Ap, Ai, x, b, residual_factprec);
      stats->relative_residual_ =
          inf_norm(n, residual_factprec) / inf_norm(n, b);
      stats->time_solve_ = get_time(time_solve);
    }
  else
    {
      // solve in factorization precision to get x0
      auto time_solve = get_time();
      ALLOC(FACTPREC, x0, n);
      copy_vector(n, b, x0);
      QDLDL_solve(Ln, Lp, Li, Lx, Dinv, x0);
      stats->time_solve_ = get_time(time_solve);

      // IR
      auto time_ir = get_time();

      double eps_work_prec = to_double(std::numeric_limits<XPREC>::epsilon());
      double thresh        = 0.5;
      double d0_norm       = to_double(std::numeric_limits<XPREC>::max());

      ALLOC(FACTPREC, correction, n);

      double b_norm = inf_norm(n, b);
      double r_norm = to_double(std::numeric_limits<WORKPREC>::max());

      copy_vector(n, x0, x);

      // IR
      for (int i = 1; i <= maxiter; i++)
        {
          ResidualCSCSymmetric<RESPREC, 0>(n, Ax, Ap, Ai, x, b, correction);
          double norm_correction = inf_norm(n, correction);
          // printf("correction: %e  x: %e \n", norm_correction, inf_norm(n,
          // x));
          scale(1. / norm_correction, n, correction);
          //  double norm_correction1 = inf_norm(n, correction);
          // printf(
          //  "correction: %e  x: %e \n\n", norm_correction1, inf_norm(n, x));
          QDLDL_solve(Ln, Lp, Li, Lx, Dinv, correction);
          scale(norm_correction, n, correction);

          double d_norm = inf_norm(n, correction);
          double x_norm = inf_norm(n, x);
          bool   stop   = (d_norm / x_norm <= 10 * eps_work_prec ||
                       d_norm / d0_norm >= thresh);

            printf("d_norm / x_norm: %e  d_norm: %e  x_norm: %e  d_norm0: %g \n", d_norm/x_norm, d_norm, x_norm, d_norm / d0_norm);

          stats->num_refinement_steps_ = i;
          // xr = xr + d
          add_vector<RESPREC>(n, correction, x, x);

          if (stop)
            {
            break;
            }
          d0_norm = d_norm;
        }

      ResidualCSCSymmetric<RESPREC, 0>(n, Ax, Ap, Ai, x, b, residual_factprec);
      r_norm                    = inf_norm(n, residual_factprec);
      stats->relative_residual_ = r_norm / b_norm;

      DEALLOC(correction);
      DEALLOC(x0);
      stats->time_ref_ = get_time(time_ir);
    }

  DEALLOC(residual_factprec);
  delete[] Li;
  delete[] Lp;
  delete[] Lx;
  delete[] D;
  delete[] Dinv;
}

#endif
