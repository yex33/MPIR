#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "tblas.hpp"
#include <vector>

template <typename PREC,
          int one_based,
          typename INT,
          typename VAL_TYPE,
          typename X_TYPE,
          typename B_TYPE,
          typename RES_TYPE>
void ResidualCSCSymmetric(INT             ncols,
                          const VAL_TYPE* values,
                          const INT*      col_ptr,
                          const INT*      row_idx,
                          const X_TYPE*   x,
                          const B_TYPE*   b,
                          RES_TYPE*       residual)
{
  std::vector<PREC> r(ncols, 0.0);

  for (auto col = 0; col < ncols; col++)
    {
      for (auto j = col_ptr[ col ] - one_based;
           j < col_ptr[ col + 1 ] - one_based;
           j++)
        {
          const PREC tmp = PREC(values[ j ]);
          const INT  row = row_idx[ j ] - one_based;

          r[ col ] += tmp * PREC(x[ row ]);

          if (row < col)
            {
              r[ row ] += tmp * PREC(x[ col ]);
            }
        }
    }

  // residual = b - r
  sub_vector<PREC>(ncols, b, r.data(), residual);
}

#endif
