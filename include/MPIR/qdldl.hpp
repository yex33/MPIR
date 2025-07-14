#ifndef QDLDL_HPP
#define QDLDL_HPP
#pragma once

#include <cstdio>
#include <limits>  //for the INT_TYPE_MAX

#define QDLDL_UNKNOWN (-1)
#define QDLDL_USED (1)
#define QDLDL_UNUSED (0)

/**
 * Compute the elimination tree for a quasidefinite matrix
 * in compressed sparse column form, where the input matrix is
 * assumed to contain data for the upper triangular part of A only,
 * and there are no duplicate indices.
 *s
 * Returns an elimination tree for the factorization A = LDL^T and a
 * count of the nonzeros in each column of L that are strictly below the
 * diagonal.
 *
 * Does not use MALLOC.  It is assumed that the arrays work, Lnz, and
 * etree will be allocated with a number of elements equal to n.
 *
 * The data in (n,Ap,Ai) are from a square matrix A in CSC format, and
 * should include the upper triangular part of A only.
 *
 * This function is only intended for factorisation of QD matrices specified
 * by their upper triangular part.  An error is returned if any column has
 * data below the diagonal or s completely empty.
 *
 * For matrices with a non-empty column but a zero on the corresponding
 * diagonal, this function will *not* return an error, as it may still be
 * possible to factor such a matrix in LDL form.   No promises are made in this
 * case though...
 *
 * @param  n      number of columns in CSC matrix A (assumed square)
 * @param  Ap     column pointers (size n+1) for columns of A
 * @param  Ai     row indices of A.  Has Ap[n] elements
 * @param  work   work vector (size n) (no meaning on return)
 * @param  Lnz    count of nonzeros in each column of L (size n) below diagonal
 * @param  etree  elimination tree (size n)
 * @return total  sum of Lnz (i.e. total nonzeros in L below diagonal).
 *                Returns -1 if the input is not triu or has an empty column.
 *                Returns -2 if the return value overflows INT.
 *
 */
template <typename INT>
INT QDLDL_etree(const INT  n,
                const INT *Ap,
                const INT *Ai,
                INT       *work,
                INT       *Lnz,
                INT       *etree) {
  INT sumLnz;
  INT i, j, p;

  for (i = 0; i < n; i++) {
    // zero out Lnz and work.  Set all etree values to unknown
    work[i]  = 0;
    Lnz[i]   = 0;
    etree[i] = QDLDL_UNKNOWN;

    // Abort if A doesn't have at least one entry
    // one entry in every column
    if (Ap[i] == Ap[i + 1]) {
      return -1;
    }
  }

  for (j = 0; j < n; j++) {
    work[j] = j;
    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];
      if (i > j) {
        printf("!!! Matrix is not upper triangular\n");
        return -1;
      };  // abort if entries on lower triangle
      while (work[i] != j) {
        if (etree[i] == QDLDL_UNKNOWN) {
          etree[i] = j;
        }
        Lnz[i]++;  // nonzeros in this column
        work[i] = j;
        i       = etree[i];
      }
    }
  }

  // compute the total nonzeros in L.  This much
  // space is required to store Li and Lx.  Return
  // error code -2 if the nonzero count will overflow
  // its unteger type.
  sumLnz = 0;
  for (i = 0; i < n; i++) {
    if (sumLnz > std::numeric_limits<INT>::max() - Lnz[i]) {
      sumLnz = -2;
      break;
    } else {
      sumLnz += Lnz[i];
    }
  }

  return sumLnz;
}

/**
 * Compute an LDL decomposition for a quasidefinite matrix
 * in compressed sparse column form, where the input matrix is
 * assumed to contain data for the upper triangular part of A only,
 * and there are no duplicate indices.
 *
 * Returns factors L, D and Dinv = 1./D.
 *
 * Does not use MALLOC.  It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx)  with sufficient space
 * allocated, with a number of nonzeros equal to the count given
 * as a return value by QDLDL_etree
 *
 * @param  n      number of columns in L and A (both square)
 * @param  Ap     column pointers (size n+1) for columns of A (not modified)
 * @param  Ai     row indices of A.  Has Ap[n] elements (not modified)
 * @param  Ax     data of A.  Has Ap[n] elements (not modified)
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  D      vectorized factor D.  Length is n
 * @param  Dinv   reciprocal of D.  Length is n
 * @param  Lnz    count of nonzeros in each column of L below diagonal,
 *                as given by QDLDL_etree (not modified)
 * @param  etree  elimination tree as as given by QDLDL_etree (not modified)
 * @param  bwork  working array of bools. Length is n
 * @param  iwork  working array of integers. Length is 3*n
 * @param  fwork  working array of PREC. Length is n
 * @return        Returns a count of the number of positive elements
 *                in D.  Returns -1 and exits immediately if any element
 *                of D evaluates exactly to zero (matrix is not quasidefinite
 *                or otherwise LDL factorisable)
 *
 * The matrix A stored in precision PREC and the factors are computed in
 * the same precision.
 *
 */

template <typename INT, typename PREC>
void mul_sub(const PREC *Lx,
             PREC        yVals_cidx,
             const INT  *Li,
             INT         beg,
             INT         end,
             PREC       *yVals) {
  int constexpr BLOCKSIZE = 8;
  INT  j                  = beg;
  PREC tmp[BLOCKSIZE];
  for (; j + BLOCKSIZE - 1 < end; j += BLOCKSIZE) {
    // #pragma clang loop vectorize(enable)
    for (int i = 0; i < BLOCKSIZE; i++) {
      tmp[i] = Lx[j + i] * yVals_cidx;
    }

    // #pragma clang loop vectorize(enable)
    for (int i = 0; i < BLOCKSIZE; i++) {
      yVals[Li[j + i]] -= tmp[i];
    }
  }

  for (; j < end; j++) {
    yVals[Li[j]] -= Lx[j] * yVals_cidx;
  }
}

template <typename INT, typename PREC>
INT QDLDL_factor(INT         n,
                 const INT  *Ap,
                 const INT  *Ai,
                 const PREC *Ax,
                 INT        *Lp,
                 INT        *Li,
                 PREC       *Lx,
                 PREC       *D,
                 PREC       *Dinv,
                 const INT  *Lnz,
                 const INT  *etree,
                 bool       *bwork,
                 INT        *iwork,
                 PREC       *fwork) {
  INT   i, k, nnzY, bidx, cidx, nextIdx, nnzE, tmpIdx;
  INT  *yIdx, *elimBuffer, *LNextSpaceInCol;
  PREC *yVals;
  PREC  yVals_cidx;
  bool *yMarkers;
  INT   positiveValuesInD = 0;

  // partition working memory into pieces
  yMarkers        = bwork;
  yIdx            = iwork;
  elimBuffer      = iwork + n;
  LNextSpaceInCol = iwork + n * 2;
  yVals           = fwork;

  Lp[0] = 0;  // first column starts at index zero

  for (i = 0; i < n; i++) {
    // compute L column indices
    Lp[i + 1] = Lp[i] + Lnz[i];  // cumsum, total at the end

    // set all Yidx to be 'unused' initially
    // in each column of L, the next available space
    // to start is just the first space in the column
    yMarkers[i]        = QDLDL_UNUSED;
    yVals[i]           = static_cast<PREC>(0.0);
    D[i]               = static_cast<PREC>(0.0);
    LNextSpaceInCol[i] = Lp[i];
  }

  // First element of the diagonal D.
  D[0] = Ax[0];
  if (D[0] == (PREC)0.0) {
    return -1;
  }
  if (D[0] > (PREC)0.0) {
    positiveValuesInD++;
  }
  Dinv[0] = (PREC)(1.) / D[0];

  // Start from 1 here. The upper LH corner is trivially 0
  // in L b/c we are only computing the subdiagonal elements
  for (k = 1; k < n; k++) {
    // NB : For each k, we compute a solution to
    // y = L(0:(k-1),0:k-1))\b, where b is the kth
    // column of A that sits above the diagonal.
    // The solution y is then the kth row of L,
    // with an implied '1' at the diagonal entry.

    // number of nonzeros in this row of L
    nnzY = 0;  // number of elements in this row

    // This loop determines where nonzeros
    // will go in the kth row of L, but doesn't
    // compute the actual values
    tmpIdx = Ap[k + 1];

    for (i = Ap[k]; i < tmpIdx; i++) {
      bidx = Ai[i];  // we are working on this element of b

      // Initialize D[k] as the element of this column
      // corresponding to the diagonal place.  Don't use
      // this element as part of the elimination step
      // that computes the k^th row of L
      if (bidx == k) {
        D[k] = Ax[i];
        continue;
      }

      yVals[bidx] = Ax[i];  // initialise y(bidx) = b(bidx)

      // use the forward elimination tree to figure
      // out which elements must be eliminated after
      // this element of b
      nextIdx = bidx;

      if (yMarkers[nextIdx] ==
          QDLDL_UNUSED) {  // this y term not already visited

        yMarkers[nextIdx] = QDLDL_USED;  // I touched this one
        elimBuffer[0] = nextIdx;  // It goes at the start of the current list
        nnzE          = 1;  // length of unvisited elimination path from here

        nextIdx = etree[bidx];

        while (nextIdx != QDLDL_UNKNOWN && nextIdx < k) {
          if (yMarkers[nextIdx] == QDLDL_USED) break;

          yMarkers[nextIdx] = QDLDL_USED;  // I touched this one
          elimBuffer[nnzE]  = nextIdx;     // It goes in the current list
          nnzE++;                          // the list is one longer than before
          nextIdx = etree[nextIdx];        // one step further along tree

        }  // end while

        // now I put the buffered elimination list into
        // my current ordering in reverse order
        while (nnzE) {
          yIdx[nnzY++] = elimBuffer[--nnzE];
        }  // end while
      }  // end if

    }  // end for i

    // This for loop places nonzeros values in the k^th row
    for (i = nnzY; i-- > 0;) {
      // which column are we working on?
      cidx = yIdx[i];

      // loop along the elements in this
      // column of L and subtract to solve to y
      tmpIdx     = LNextSpaceInCol[cidx];
      yVals_cidx = yVals[cidx];

// NN This is the most time consuming part. Below is rewritten.
// #define ORIGINAL
#ifdef ORIGINAL
      for (int j = Lp[cidx]; j < tmpIdx; j++) {
        yVals[Li[j]] -= Lx[j] * yVals_cidx;
      }
#else
      mul_sub(Lx, yVals_cidx, Li, Lp[cidx], tmpIdx, yVals);

#endif

      // Now I have the cidx^th element of y = L\b.
      // so compute the corresponding element of
      // this row of L and put it into the right place
      Li[tmpIdx] = k;
      Lx[tmpIdx] = yVals_cidx * Dinv[cidx];

      // D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
      D[k] -= yVals_cidx * Lx[tmpIdx];
      LNextSpaceInCol[cidx]++;

      // reset the yvalues and indices back to zero and QDLDL_UNUSED
      // once I'm done with them
      yVals[cidx]    = static_cast<PREC>(0.0);
      yMarkers[cidx] = QDLDL_UNUSED;

    }  // end for i

    // Maintain a count of the positive entries
    // in D.  If we hit a zero, we can't factor
    // this matrix, so abort
    if (D[k] == PREC(0.0)) {
      return -1;
    }
    if (D[k] > 0.0) {
      positiveValuesInD++;
    }

    // compute the inverse of the diagonal
    Dinv[k] = PREC(1.) / D[k];
    // printf("ddd %e  %e\n", double(D[k]), double(Dinv[k]));

  }  // end for k

  return positiveValuesInD;
}

/**
 * Solves (L+I)x = b
 *
 * It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx).
 *
 * @param  n      number of columns in L
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  x      initialize to b.  Equal to x on return
 *
 * The L factor is in factorization precision FACTPREC. x is in the precision
 * in which we compute the results.
 */
template <typename INT, typename FACTPREC, typename XPREC>
void QDLDL_Lsolve(
    INT n, const INT *Lp, const INT *Li, const FACTPREC *Lx, XPREC *x) {
  INT i, j;
  for (i = 0; i < n; i++) {
    XPREC val = x[i];  // store x[i] into  val of precision XPREC.
    for (j = Lp[i]; j < Lp[i + 1]; j++) {
      x[Li[j]] -= XPREC(Lx[j]) * val;
    }
  }
}

/**
 * Solves (L+I)'x = b
 *
 * It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx).
 *
 * @param  n      number of columns in L
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  x      initialized to b.  Equal to x on return
 *
 * Lx is in factoriztion precision. The intermediate results and x are computed
 * in the precision of x.
 */
template <typename INT, typename FACTPREC, typename XPREC>
void QDLDL_Ltsolve(
    INT n, const INT *Lp, const INT *Li, const FACTPREC *Lx, XPREC *x) {
  INT i, j;
  for (i = n; i-- > 0;) {
    XPREC val = x[i];
    for (j = Lp[i]; j < Lp[i + 1]; j++) {
      val -= XPREC(Lx[j]) * x[Li[j]];
    }
    x[i] = val;
  }
}

/**
 * Solves LDL'x = b
 *
 * It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx).
 *
 * @param  n      number of columns in L
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  Dinv   reciprocal of D.  Length is n
 * @param  x      initialized to b.  Equal to x on return
 *
 * The factors are in FACT_PREC. The solving is done in X_PREC.
 */
template <typename INT, typename FACTPREC, typename XPREC>
void QDLDL_solve(const INT       n,
                 const INT      *Lp,
                 const INT      *Li,
                 const FACTPREC *Lx,
                 const FACTPREC *Dinv,
                 XPREC          *x) {
  INT i;

  // Solves Ax = LDL' x = b
  // Let y = DL'x.
  QDLDL_Lsolve(n, Lp, Li, Lx, x);  // Solve Ay = b
  // Once we find y, find L'x = D^(-1)y
  for (i = 0; i < n; i++) x[i] *= XPREC(Dinv[i]);  // D^(-1)y
  QDLDL_Ltsolve(n, Lp, Li, Lx, x);                 // Solves L'x = D^(-1)y
}

// End of QDLDL_HPP
#endif
