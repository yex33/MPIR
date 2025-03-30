#include "gmres_ir.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "qdldl.hpp"

template <typename U_F, typename U_W, typename U_R>
void GMRESLDLIR<U_F, U_W, U_R>::compute(std::size_t                     n,
                                        const std::vector<std::size_t>& Ap,
                                        const std::vector<std::size_t>& Ai,
                                        const std::vector<U_W>&         Ax) {
  // TODO: move semantics?
  this->n  = n;
  this->Ap = Ap;
  this->Ai = Ai;
  this->Ax = Ax;

  std::vector<std::size_t> iwork(3 * n);
  std::vector<std::size_t> Lcolnz(n);
  std::vector<std::size_t> etree(n);

  std::size_t Lnz = QDLDL_etree(n, Ap.data(), Ai.data(), iwork.data(),
                                Lcolnz.data(), etree.data());

  Lp.resize(n + 1);
  Li.resize(Lnz);
  Lx.resize(Lnz);

  std::vector<U_F> D(n);
  std::vector<U_F> Dinv(n);
  bool*            bwork = new bool[n];
  std::vector<U_F> fwork(n);
  if constexpr (std::is_same_v<U_F, U_W>) {
    QDLDL_factor(n, Ap.data(), Ai.data(), Ax.data(), Lp.data(), Li.data(),
                 Lx.data(), D.data(), Dinv.data(), Lcolnz.data(), etree.data(),
                 bwork, iwork.data(), fwork.data());
  } else {
    std::vector<U_F> Ax_U_F(Ax.size());
    std::transform(Ax.cbegin(), Ax.cend(), Ax_U_F.begin(),
                   [](U_W x) { return static_cast<U_F>(x); });
    QDLDL_factor(n, Ap.data(), Ai.data(), Ax_U_F.data(), Lp.data(), Li.data(),
                 Lx.data(), D.data(), Dinv.data(), Lcolnz.data(), etree.data(),
                 bwork, iwork.data(), fwork.data());
  }

  delete[] bwork;
}
