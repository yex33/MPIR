#ifndef GMRES_IR_HPP
#define GMRES_IR_HPP
#pragma once

#include <cstddef>
#include <vector>

template <typename U_F, typename U_W, typename U_R>
class GMRESLDLIR {
 private:
  std::size_t              n;
  std::vector<std::size_t> Ap;
  std::vector<std::size_t> Ai;
  std::vector<U_W>         Ax;

  std::vector<std::size_t> Lp;
  std::vector<std::size_t> Li;
  std::vector<U_F>         Lx;

 public:
  void compute(std::size_t                     n,
               const std::vector<std::size_t>& Ap,
               const std::vector<std::size_t>& Ai,
               const std::vector<U_W>&         Ax);
};

// End of GMRES_IR_HPP
#endif
