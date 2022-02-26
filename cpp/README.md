# `phase_space`

## Lorentz Vector

Create a 4-vector with energy 15, and three-momentum components (1.0, 2.0, 3.0):

```c++
phase_space::LVector<double> p = {15.0, 1.0, 2.0, 3.0};
```

Access elements of the 4-vector:

```cpp
double energy = p.e();
double px = p.px();
double py = p.py();
double pz = p.pz();
```

Set elements:

```cpp
p.e() = 20.0;
p.px() = 1.0;
p.py() = 2.0;
p.pz() = 3.0;
std::cout << p << "\n"; // You can print to cout
```

Compute norms:

```cpp
// Compute the 'mass' (just sqrt(p.p))
auto m = phase_space::lnorm(p);
// Compute the squared 'mass' (p.p)
auto m2 = phase_space::lnorm_sqr(p);
```

Binary operations

```cpp
phase_space::LVector<double> p = {15.0, 1.0, 2.0, 3.0};
phase_space::LVector<double> q = {10.0, 0.0, 0.0, 1.0};
std::cout << p + q << "\n";        // add them
std::cout << p - q << "\n";        // subtract them
std::cout << 10 * p << "\n";       // multiply by scalar
auto dot = phase_space::dot(p, q); // scalar product
```

## Compute decay width or cross section

There is a function to compute the decay-width `phase_space::decay_width` and
another to compute the cross section `phase_space::cross_section`. These have
the following signatures:

```cpp
double msqrd(const std::array<phase_space::LVector<double>, N>& momenta) {
    // Compute squared matrix element
}

auto result = phase_space::decay_width(
    msqrd,
    m,
    fsp_masses,
    nevents=10'000,
    batchsize=100
);
```

where

- `msqrd`: Function taking in an array of the final state four-momenta and
  returns the spin-averaged squared matrix element
- `m`: Mass of the decaying particle
- `fsp_masses`: an `std::array` containing the masses of the final state
  particles
- `nevents`: Number of points used in the Monte-Carlo integration. Increasing
  will increase the accurary of the result.
- `batchsize`: Number of points to compute per thread (probably don't need
  to change)
- `result`: Returns the estimated decay width and the estimates error.

For the cross-section:

```cpp
auto result = phase_space::cross_section(
    msqrd,
    cme,
    m1,
    m2,
    fsp_masses,
    nevents=10'000,
    batchsize=100
);
```

- `msqrd`: (same as `decay_width`)
- `cme`: Center-of-mass energy
- `m1`: Mass of 1st incoming particle
- `m2`: Mass of 2nd incoming particle
- `fsp_masses`: (same as `decay_width`)
- `nevents`: (same as `decay_width`)
- `batchsize`: (same as `decay_width`)
- `result`: (same as `decay_width`)

### Example: Muon decay

Compute the Muon decay width into an electron and two neutrinos:

```cpp
#include "phase_space.hpp"

static constexpr double MUON_MASS = 105.6583745e-3;
static constexpr double G_FERMI = 1.1663787e-5;

auto msqrd_mu_to_e_nu_nu(const phase_space::MomentaType<3> &ps) -> double {
  using phase_space::lnorm_sqr;
  using phase_space::tools::sqr; // sqr(x) is identical to x * x
  const auto t = phase_space::lnorm_sqr(ps[0] + ps[2]);
  constexpr double mmu = MUON_MASS;
  return 16 * sqr(G_FERMI) * t * (sqr(mmu) - t);
}

int main() {
  using phase_space::decay_width;
  using phase_space::tools::powi; // powi<n>(x) is identical to x * x ... * x n times
  using phase_space::tools::sqr;  // sqr(x) is identical to x * x

  const std::array<double, 3> fsp_masses{0.0, 0.0, 0.0};
  const size_t nevents = 100'000;

  const double width = sqr(G_FERMI) * powi<5>(MUON_MASS) / (192 * pow(M_PI, 3));
  const auto res = decay_width(msqrd_mu_to_e_nu_nu, MUON_MASS, fsp_masses, nevents);
  const auto frac_diff = std::abs((width - res.first) / width);

  std::cout << "estimate = " << res.first << " +- " << res.second << std::endl;
  std::cout << "actual = " << width << std::endl;
  std::cout << "frac. diff. (%) = " << frac_diff * 100.0 << std::endl;
  return 0.0;
}
```

Compile and run:

```shell
clang++ -I/dir/containing/header -fPIC -march=native -O3 test_muon_decay.cpp -o test_muon_decay.out
./test_muon_decay.out
```

or use cmake:

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOPTIMIZE_FOR_NATIVE=1 ..
make
./test_muon_decay
```

This will print something like:

    estimate = 3.01022e-19 +- 4.26305e-22
    actual = 3.00918e-19
    frac. diff. (%) = 0.0345131

### Example: 2 -> 4 Thermal cross section

```cpp
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <gsl/gsl_sf_bessel.h>
#include <phase_space.hpp>

using MomentaType = std::array<phase_space::LVector<double>, 4>;

/// Struct holding whatever parameters are needed
struct ModelParameters{
    double mx;
    std::array<double, 4> fsp_masses;
};

/// Function computing squared matrix element
auto msqrd(const MomentaType& momenta, const ModelParameters& params) -> double {
    return 0.0;
}

/// Compute zero-temperature cross section
auto cross_section(double cme, const ModelParameters& params) -> double {
    auto msqrd_ = [&params](const MomentaType& momenta){
        return msqrd(momenta, params);
    };
    return phase_space::cross_section(msqrd_, cme, params.mx, params.mx, params.fsp_masses).first;
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
        const double ker = z2 * (z2 - 4.0) * gsl_sf_bessel_K1_scaled(x * z) * exp(-x * (z - 2.0));
        return sig * ker;
    };

    const double integral = gauss_kronrod<double, 15>::integrate(
        f, 4.0, std::numeric_limits<double>::infinity(), 5, 1e-8);

    return pre * integral;
}
```
