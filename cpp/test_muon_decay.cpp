#include "phase_space.hpp"

static constexpr double MUON_MASS = 105.6583745e-3;
static constexpr double G_FERMI = 1.1663787e-5;

// Function to compute the spin-average squared matrix element for mu -> e + nu
// + nu Here `phase_space::MomentaType<3>` is an std::array of
// `phase_space::LVector<double>`
auto msqrd_mu_to_e_nu_nu(const phase_space::MomentaType<3> &ps) -> double {
  using phase_space::lnorm_sqr;
  using phase_space::tools::sqr; // sqr(x) is identical to x * x

  const auto t = phase_space::lnorm_sqr(ps[0] + ps[2]);
  constexpr double mmu = MUON_MASS;
  return 16 * sqr(G_FERMI) * t * (sqr(mmu) - t);
}

int main() {
  using phase_space::decay_width;
  // powi<n>(x) is identical to x * x ... * x n times
  using phase_space::tools::powi;
  // sqr(x) is identical to x * x
  using phase_space::tools::sqr;

  const std::array<double, 3> fsp_masses{0.0, 0.0, 0.0};
  const size_t nevents = 100'000;

  const double width = sqr(G_FERMI) * powi<5>(MUON_MASS) / (192 * pow(M_PI, 3));
  const auto res =
      decay_width(msqrd_mu_to_e_nu_nu, MUON_MASS, fsp_masses, nevents);
  const auto frac_diff = std::abs((width - res.first) / width);

  std::cout << "estimate = " << res.first << " +- " << res.second << std::endl;
  std::cout << "actual = " << width << std::endl;
  std::cout << "frac. diff. (%) = " << frac_diff * 100.0 << std::endl;
}
