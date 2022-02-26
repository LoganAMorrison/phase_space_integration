#include "phase_space.hpp"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <gsl/gsl_sf_bessel.h>

using MomentaType = std::array<phase_space::LVector<double>, 4>;

/// Struct holding whatever parameters are needed
struct ModelParameters {
  double mx;
  std::array<double, 4> fsp_masses;
};

/// Function computing squared matrix element
auto msqrd(const MomentaType &momenta, const ModelParameters &params)
    -> double {
  return 1.0;
}

/// Compute zero-temperature cross section
auto cross_section(double cme, const ModelParameters &params) -> double {
  auto msqrd_ = [&params](const MomentaType &momenta) {
    return msqrd(momenta, params);
  };
  return phase_space::cross_section(msqrd_, cme, params.mx, params.mx,
                                    params.fsp_masses)
      .first;
}

/// Compute the thermal cross section for X + X -> A + B + C + D
///
/// Usual integal is:
///     <σv> = (2 π² T) ∫ ds σ[√s] (s - 4 m²) √s K₁[√s/T] / (4π m² T K₂[m/T])
/// Define: s = z² m², x = m /T. Then this becomes:
///     <σv> = x ∫ dz z² (z² - 4) σ[m z] K₁[x * z] / (4 K₂[x]²)
/// Next, define scaled bessel functions: K₁[x] = exp(-x) k₁[x]
/// and K₂[x] = exp(-x) k₂[x]. Then:
///     <σv> = x / (4 k₂[x]²) ∫ dz z² (z² - 4) σ[m z] k₁[x * z] exp(-x (z-2))
double thermal_cross_section(const double x, const ModelParameters &params) {
  using boost::math::quadrature::gauss_kronrod;

  const double den = 2.0 * gsl_sf_bessel_Kn_scaled(2, x);
  const double pre = x / (den * den);
  const double mx = params.mx;

  auto f = [x, &params, mx](double z) -> double {
    const double z2 = z * z;
    const double sig = cross_section(z * mx, params);
    const double ker =
        z2 * (z2 - 4.0) * gsl_sf_bessel_K1_scaled(x * z) * exp(-x * (z - 2.0));
    return sig * ker;
  };

  const double integral = gauss_kronrod<double, 15>::integrate(
      f, 4.0, std::numeric_limits<double>::infinity(), 5, 1e-8);

  return pre * integral;
}

int main() {

  ModelParameters params = {100.0, {1.0, 1.0, 1.0, 1.0}};
  std::cout << thermal_cross_section(1.0, params);
  return 0.0;
}
