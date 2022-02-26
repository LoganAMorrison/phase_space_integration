#ifndef PHASE_SPACE_HPP
#define PHASE_SPACE_HPP

#include <array>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

namespace phase_space {

// ==========================================================================
// ---- Useful Compile Time Functions ---------------------------------------
// ==========================================================================

namespace tools {

/// Compute the square of a number.
template <typename T> constexpr auto sqr(const T &x) -> decltype(x * x) {
  return x * x;
}

/// Compute the absolute square of a number.
template <typename T>
inline constexpr auto abs2(const T &x) -> decltype(std::abs(x) * std::abs(x)) {
  return std::abs(x) * std::abs(x);
}

// =========================================
// ---- Compile-time non-negative power ----
// =========================================

// Even power
template <unsigned int N, unsigned int M = N % 2> struct positive_power {
  template <typename T> static auto result(T base) -> T {
    T power = positive_power<N / 2>::result(base);
    return power * power;
  }
};

// Odd power
template <unsigned int N> struct positive_power<N, 1> {
  template <typename T> static auto result(T base) -> T {
    T power = positive_power<N / 2>::result(base);
    return base * power * power;
  }
};

template <> struct positive_power<1, 1> { // NOLINT
  template <typename T> static auto result(T base) -> T { return base; }
};

template <> struct positive_power<0, 0> { // NOLINT
  template <typename T> static auto result(T /*base*/) -> T { return T(1); }
};

// Compute base^p.
template <unsigned int N, typename T> static auto powi(T base) -> T {
  return positive_power<N>::result(base);
}

// ================================
// ---- Compile-Time Factorial ----
// ================================

template <unsigned int N> struct factorial {
  static auto result() -> unsigned int {
    return factorial<N - 1>::result() * N;
  }
};

template <> struct factorial<1> { // NOLINT
  static auto result() -> unsigned int { return 1; }
};

template <> struct factorial<0> { // NOLINT
  static auto result() -> unsigned int { return 1; }
};

template <unsigned int N> auto fact() -> unsigned int {
  return factorial<N>::result();
}

// =====================================================
// ---- Statistics Functions ---------------------------
// =====================================================

static auto mean_sum_sqrs_welford(const std::vector<double> &wgts)
    -> std::tuple<double, double, double> {

  double mean = 0.0;
  double m2 = 0.0;
  double count = 0.0;

  for (const auto &wgt : wgts) {
    count += 1;
    const double delta = wgt - mean;
    mean += delta / count;
    const double delta2 = wgt - mean;
    m2 += delta * delta2;
  }

  return std::make_tuple(mean, m2, count);
}

static auto mean_var_welford(const std::vector<double> &wgts)
    -> std::pair<double, double> {

  double mean = 0.0;
  double m2 = 0.0;
  double count = 0.0;

  for (const auto &wgt : wgts) {
    count += 1;
    const double delta = wgt - mean;
    mean += delta / count;
    const double delta2 = wgt - mean;
    m2 += delta * delta2;
  }

  return std::make_pair(mean, m2 / (count - 1));
}

static auto mean_var(const std::vector<double> &wgts)
    -> std::pair<double, double> {
  const double inv_n = 1.0 / static_cast<double>(wgts.size());

  const double mean = inv_n * std::accumulate(wgts.begin(), wgts.end(), 0.0);

  auto var_reduce = [mean](double a, double b) { return a + sqr(b - mean); };
  const double var =
      inv_n * std::accumulate(wgts.begin(), wgts.end(), 0.0, var_reduce);

  return std::make_pair(mean, var);
}

// =========================================================
// ---- Kinematics -----------------------------------------
// =========================================================

template <typename T>
static inline auto kallen_lambda(const T a, const T b, const T c) -> T {
  return std::fma(a, a - 2 * c, std::fma(b, b - 2 * a, c * (c - 2 * b)));
}

} // namespace tools

// ==========================================================================
// ---- Lorentz Vectors -----------------------------------------------------
// ==========================================================================

template <typename T> class LVector {
private:
  std::array<T, 4> p_data{};

public:
  explicit LVector(std::array<T, 4> data) : p_data(std::move(data)) {}
  LVector(T x0, T x1, T x2, T x3) : p_data({x0, x1, x2, x3}) {}
  LVector() = default;

  // ---- Access Operations ---------------------------------------------------

  auto e() const -> const T & { return p_data[0]; }
  auto e() -> T & { return p_data[0]; }

  auto px() const -> const T & { return p_data[1]; }
  auto px() -> T & { return p_data[1]; }

  auto py() const -> const T & { return p_data[2]; }
  auto py() -> T & { return p_data[2]; }

  auto pz() const -> const T & { return p_data[3]; }
  auto pz() -> T & { return p_data[3]; }

  auto operator[](size_t i) const -> const T & { return p_data[i]; }
  auto operator[](size_t i) -> T & { return p_data[i]; }

  auto at(size_t i) const -> const T & { return p_data.at(i); }
  auto at(size_t i) -> T & { return p_data.at(i); }

  auto data() const noexcept -> const T * { return p_data.data(); }
  auto data() noexcept -> T * { return p_data.data(); }

  // ---- Printing Operations -------------------------------------------------

  friend auto operator<<(std::ostream &os, const LVector<T> &p)
      -> std::ostream & {
    os << "LVector(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3]
       << ")";
    return os;
  }

  // ---- Unary Math Operations -----------------------------------------------

  auto operator-() const -> LVector<decltype(-T{})> {
    return LVector(-p_data[0], -p_data[1], -p_data[2], -p_data[3]);
  }

  // ---- Non-Modifying Binary Math Operations --------------------------------

  template <typename S>
  auto operator+(const LVector<S> &rhs) const -> LVector<decltype(T{} + S{})> {
    return LVector(p_data[0] + rhs.p_data[0], p_data[1] + rhs.p_data[1],
                   p_data[2] + rhs.p_data[2], p_data[3] + rhs.p_data[3]);
  }

  template <typename S>
  auto operator-(const LVector<S> &rhs) const -> LVector<decltype(T{} - S{})> {
    return LVector(p_data[0] - rhs.p_data[0], p_data[1] - rhs.p_data[1],
                   p_data[2] - rhs.p_data[2], p_data[3] - rhs.p_data[3]);
  }

  // ---- Modifying Binary Math Operations ------------------------------------

  template <typename S> auto operator+=(const LVector<S> &rhs) -> void {
    p_data[0] += static_cast<T>(rhs.p_data[0]);
    p_data[1] += static_cast<T>(rhs.p_data[1]);
    p_data[2] += static_cast<T>(rhs.p_data[2]);
    p_data[3] += static_cast<T>(rhs.p_data[3]);
  }

  template <typename S> auto operator-=(const LVector<S> &rhs) -> void {
    p_data[0] -= static_cast<T>(rhs.p_data[0]);
    p_data[1] -= static_cast<T>(rhs.p_data[1]);
    p_data[2] -= static_cast<T>(rhs.p_data[2]);
    p_data[3] -= static_cast<T>(rhs.p_data[3]);
  }

  template <typename S> auto operator*=(const S &rhs) -> void {
    p_data[0] *= static_cast<T>(rhs);
    p_data[1] *= static_cast<T>(rhs);
    p_data[2] *= static_cast<T>(rhs);
    p_data[3] *= static_cast<T>(rhs);
  }

  template <typename S> auto operator/=(const S &rhs) -> void {
    p_data[0] /= static_cast<T>(rhs);
    p_data[1] /= static_cast<T>(rhs);
    p_data[2] /= static_cast<T>(rhs);
    p_data[3] /= static_cast<T>(rhs);
  }

  // ---- Friend Math Operations ----------------------------------------------

  template <typename U>
  friend auto operator*(const LVector<T> &p, const U &rhs)
      -> LVector<decltype(U{} * T{})> {
    return LVector<decltype(U{} * T{})>(p[0] * rhs, p[1] * rhs, p[2] * rhs,
                                        p[3] * rhs);
  }

  template <typename U>
  friend auto operator*(const U &rhs, const LVector<T> &p)
      -> LVector<decltype(U{} * T{})> {
    return p * rhs;
  }

  template <typename U>
  friend auto operator/(const LVector<T> &p, const U &rhs)
      -> LVector<decltype(T{} / U{})> {
    return LVector<decltype(T{} / U{})>(p[0] / rhs, p[1] / rhs, p[2] / rhs,
                                        p[3] / rhs);
  }
};

// ==========================================================================
// ---- Norms ---------------------------------------------------------------
// ==========================================================================

template <typename T>
inline auto lnorm_sqr(const LVector<T> &lv) -> decltype(tools::abs2(T{})) {
  using tools::abs2;
  return abs2(lv.e()) - (abs2(lv.px()) + abs2(lv.py()) + abs2(lv.pz()));
}

template <typename T>
inline auto lnorm(const LVector<T> &lv) -> decltype(sqrt(std::abs(T{} * T{}))) {
  const auto m2 = lnorm_sqr(lv);
  return sqrt(std::abs(m2));
}

template <typename T>
inline auto lnorm3_sqr(const LVector<T> &lv) -> decltype(tools::abs2(T{})) {
  using tools::abs2;
  return abs2(lv.px()) + abs2(lv.py()) + abs2(lv.pz());
}

template <typename T>
inline auto lnorm3(const LVector<T> &lv)
    -> decltype(sqrt(std::abs(T{} * T{}))) {
  return sqrt(lnorm3_sqr(lv));
}

// ==========================================================================
// ---- Operations on LVectors ----------------------------------------------
// ==========================================================================

template <typename T, typename S>
inline auto dot(const LVector<T> &lv1, const LVector<S> &lv2)
    -> decltype(T{} * S{}) {
  return lv1[0] * lv2[0] -
         (lv1[1] * lv2[1] + lv1[2] * lv2[2] + lv1[3] * lv2[3]);
}

// ==========================================================================
// ---- Aliases -------------------------------------------------------------
// ==========================================================================

using MomentumType = LVector<double>;
template <size_t N> using MomentaType = std::array<MomentumType, N>;

namespace detail {

static constexpr size_t DEFAULT_NEVENTS = 10000;
static constexpr size_t DEFAULT_BATCHSIZE = 100;

// =====================================================
// ---- Checks -----------------------------------------
// =====================================================

template <size_t N> static inline auto check_nfsp() -> void {
  static_assert(N >= 2, "Number of final-state particles must be >= 2.");
}

template <size_t N>
static inline auto channel_open(double cme, const std::array<double, N> &masses)
    -> bool {
  const double msum = std::accumulate(masses.begin(), masses.end(), 0.0);
  return cme > msum;
}

// =========================================
// ---- Core Algorithm Functions -----------
// =========================================

template <size_t N> auto massless_weight(double cme) -> double {
  using tools::factorial;
  using tools::powi;
  using tools::sqr;

  static_assert(N >= 2, "Number of final-state particles must be >= 2.");

  static constexpr double k1_2PI = 1.591'549'430'918'953'5e-1;
  static constexpr double PRE_2 = 3.978'873'577'297'383'6e-2;
  static constexpr double PRE_3 = 1.259'825'563'796'855e-4;
  static constexpr double PRE_4 = 1.329'656'430'278'884e-7;
  static constexpr double PRE_5 = 7.016'789'757'994'902e-11;

  switch (N) {
  case 2:
    return PRE_2;
  case 3:
    return PRE_3 * sqr(cme);
  case 4:
    return PRE_4 * powi<4>(cme);
  case 5:
    return PRE_5 * powi<6>(cme);
  default:
    break;
  }

  const double wgt =
      powi<N - 1>(M_PI_2) * powi<2 * N - 4>(cme) * powi<3 * N - 4>(k1_2PI);

  // Compute (n-1)! * (n-2)!
  const auto fact_nm2 = factorial<N - 2>::result();
  const double fact = 1.0 / static_cast<double>((N - 1) * fact_nm2 * fact_nm2);
  return fact * wgt;
}

// Generate a `n` random four-momenta with energies distributated according to
// E⋅exp(-E)
template <size_t N> auto initialize_momenta(MomentaType<N> *momenta) -> void {
  static thread_local std::random_device rd{};
  static thread_local std::mt19937 generator{rd()};
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // 2*pi
  static constexpr double k2PI = 6.2831853071795865;

#pragma unroll N
  for (auto &p : *momenta) {
    const double rho1 = distribution(generator);
    const double rho2 = distribution(generator);
    const double rho3 = distribution(generator);
    const double rho4 = distribution(generator);

    const double ctheta = 2 * rho1 - 1;
    const double stheta = sqrt(1 - ctheta * ctheta);
    const double phi = k2PI * rho2;
    const double e = -log(rho3 * rho4);

    p[0] = e;
    p[1] = e * stheta * cos(phi);
    p[2] = e * stheta * sin(phi);
    p[3] = e * ctheta;
  }
}

template <size_t N>
auto boost_momenta(MomentaType<N> *momenta, double cme) -> void {
  MomentumType sum_qs = std::accumulate(momenta->begin(), momenta->end(),
                                        LVector<double>{0.0, 0.0, 0.0, 0.0});

  for (const auto &p : *momenta) {
    sum_qs[0] += p[0];
    sum_qs[1] += p[1];
    sum_qs[2] += p[2];
    sum_qs[3] += p[3];
  }

  const double invmass = 1.0 / lnorm(sum_qs);
  // boost vector
  const double bx = -invmass * sum_qs[1];
  const double by = -invmass * sum_qs[2];
  const double bz = -invmass * sum_qs[3];
  // boost factors
  const double x = cme * invmass;
  const double g = sum_qs[0] * invmass;
  const double a = 1.0 / (1.0 + g);

#pragma unroll N
  for (auto &p : *momenta) {
    const double bdotq = bx * p[1] + by * p[2] + bz * p[3];
    const double fact = std::fma(a, bdotq, p[0]);

    p[0] = x * std::fma(g, p[0], bdotq);
    p[1] = x * std::fma(fact, bx, p[1]);
    p[2] = x * std::fma(fact, by, p[2]);
    p[3] = x * std::fma(fact, bz, p[3]);
  }
}

template <size_t N>
auto compute_scale_factor(const MomentaType<N> &ps, double cme,
                          const std::array<double, N> &ms) -> double {
  static constexpr size_t max_iter = 50;
  static constexpr double tol = 1e-10;
  // initial guess
  const double msum = std::accumulate(ms.begin(), ms.end(), 0);
  double xi = sqrt(1.0 - tools::sqr(msum / cme));

  size_t itercount = 0;
  while (true) {
    double f = -cme;
    double df = 0.0;
    // Compute residual and derivative
#pragma unroll N
    for (size_t i = 0; i < N; i++) {
      const double e = ps[i][0];
      const double deltaf = std::hypot(ms[i], xi * e);
      f += deltaf;
      df += xi * tools::sqr(e) / deltaf;
    }

    // Newton correction
    const double deltaxi = -f / df;
    xi += deltaxi;

    itercount += 1;
    if (std::abs(deltaxi) < tol || itercount > max_iter) {
      break;
    }
  }
  return xi;
}

template <size_t N>
auto correct_masses(MomentaType<N> *ps, double cme,
                    const std::array<double, N> &ms) -> void {
  const double xi = compute_scale_factor(*ps, cme, ms);
#pragma unroll N
  for (size_t i = 0; i < N; i++) {
    ps->at(i)[0] = std::hypot(ms[i], xi * ps->at(i)[0]);
    ps->at(i)[1] *= xi;
    ps->at(i)[2] *= xi;
    ps->at(i)[3] *= xi;
  }
}

template <size_t N>
auto wgt_rescale_factor(const MomentaType<N> &ps, double cme) -> double {
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 1.0;
#pragma unroll N
  for (const auto &p : ps) {
    const double modsqr = lnorm3_sqr(p);
    const double mod = sqrt(modsqr);
    const double inveng = 1.0 / p[0];
    t1 += mod / cme;
    t2 += modsqr * inveng;
    t3 *= mod * inveng;
  }
  t1 = pow(t1, static_cast<double>(2 * N - 3));
  t2 = 1.0 / t2;
  return t1 * t2 * t3 * cme;
}

// Fill momenta in center-of-mass frame with center-of-mass energy `cme`
// and `masses`.
template <size_t N>
auto generate_momenta(MomentaType<N> *momenta, double cme,
                      const std::array<double, N> &ms) -> void {
  initialize_momenta(momenta);
  boost_momenta(momenta, cme);
  correct_masses(momenta, cme, ms);
}

template <size_t N, class MSqrd>
auto generate_wgt(MSqrd msqrd, MomentaType<N> *momenta, double cme,
                  const std::array<double, N> &ms, double base_wgt) -> double {
  generate_momenta(momenta, cme, ms);
  return wgt_rescale_factor(*momenta, cme) * msqrd(*momenta) * base_wgt;
}

} // namespace detail

/**
 * Integrate over an N-particle phase-space.
 *
 * @param msqrd Function to compute squared matrix element given the momenta
 * @param cme Center-of-mass energy
 * @param masses Masses of the final-state particles
 * @param nevents Number of phase-space points to sample
 * @param batchsize Number of phase-space points process at a time
 */
template <class MSqrd, size_t N>
static auto
integrate_phase_space(MSqrd msqrd, double cme, const std::array<double, N> &ms,
                      const size_t nevents = detail::DEFAULT_NEVENTS,
                      const size_t batchsize = detail::DEFAULT_BATCHSIZE)
    -> std::pair<double, double> {
  const double base_wgt = detail::massless_weight<N>(cme);
  const double inv_nevents = 1.0 / static_cast<double>(nevents);

  double mean = 0.0;
  double m2 = 0.0;
  double count = 0.0;

  std::mutex mutex;

  const auto adder = [&mean, &m2, &count, &mutex, &cme, &ms, &base_wgt,
                      &msqrd](size_t n) {
    std::vector<double> weights(n);
    MomentaType<N> momenta{};

    // generate 'local' events
    for (size_t i = 0; i < n; i++) { // NOLINT
      weights[i] = detail::generate_wgt(msqrd, &momenta, cme, ms, base_wgt);
    }
    // Compute mean and sum of squares
    auto lmv = tools::mean_sum_sqrs_welford(weights);

    // Lock access and add results to 'global' values
    std::lock_guard<std::mutex> g(mutex);
    const double avgb = std::get<0>(lmv);
    const double m2b = std::get<1>(lmv);
    const double nb = std::get<2>(lmv);

    const double na = count;
    count += nb;

    const double delta = mean - avgb;

    mean = (na * mean + nb * avgb) / count;
    m2 += m2b + tools::sqr(delta) * na * nb / count;
  };

  // Compute the number of different threads to launch
  size_t nbatches = nevents / batchsize;
  const size_t remaining = nevents % batchsize;

  // Create threads and launch, then wait for them to finish
  std::vector<std::thread> threads;
  threads.reserve(nbatches + (remaining > 0 ? 1 : 0));

  for (size_t i = 0; i < nbatches; i++) { // NOLINT
    threads.emplace_back(adder, batchsize);
  }
  if (remaining > 0) {
    threads.emplace_back(adder, remaining);
  }
  for (auto &t : threads) { // NOLINT
    t.join();
  }

  // Compute global mean and std
  const double var = m2 / static_cast<double>(count - 1);
  const double std = sqrt(var * inv_nevents);

  return std::make_pair(mean, std);
}

/// Compute the decay width of a particle.
///
/// @param msqrd Function to compute squared matrix element given the momenta
/// @param m Mass of the decaying particle
/// @param masses Masses of the final-state particles
/// @param nevents Number of phase-space points to sample
/// @param batchsize Number of phase-space points process per thread
template <size_t N, class MSqrd>
static auto decay_width(MSqrd msqrd, const double m,
                        const std::array<double, N> &masses,
                        const size_t nevents = detail::DEFAULT_NEVENTS,
                        const size_t batchsize = detail::DEFAULT_BATCHSIZE)
    -> std::pair<double, double> {
  detail::check_nfsp<N>();
  if (!detail::channel_open(m, masses)) {
    return std::make_pair(0.0, 0.0);
  }
  auto result = integrate_phase_space(msqrd, m, masses, nevents, batchsize);
  result.first = result.first / (2.0 * m);
  result.second = result.second / (2.0 * m);
  return result;
}

/// Compute the decay width or scattering cross-section.
///
/// @param msqrd Function to compute squared matrix element given the momenta
/// @param cme Center-of-mass energy
/// @param m1 Mass of 1st incoming particle
/// @param m2 Mass of 2nd incoming particle
/// @param masses Masses of the final-state particles
/// @param nevents Number of phase-space points to sample
/// @param batchsize Number of phase-space points process per thread
template <size_t N, class MSqrd>
static auto cross_section(MSqrd msqrd, const double cme, double m1, double m2,
                          const std::array<double, N> &masses,
                          const size_t nevents = detail::DEFAULT_NEVENTS,
                          const size_t batchsize = detail::DEFAULT_BATCHSIZE)
    -> std::pair<double, double> {
  detail::check_nfsp<N>();
  if (!detail::channel_open(cme, masses) || m1 + m2 > cme) {
    return std::make_pair(0.0, 0.0);
  }

  auto res = integrate_phase_space(msqrd, cme, masses, nevents, batchsize);

  const auto mu1 = tools::sqr(m1 / cme);
  const auto mu2 = tools::sqr(m2 / cme);
  // (2 * E1) * (2 * E2) * vrel
  const auto den =
      2.0 * cme * cme * std::sqrt(tools::kallen_lambda(1.0, mu1, mu2));

  res.first /= den;
  res.second /= den;
  return res;
}

} // namespace phase_space

#endif // PHASE_SPACE_HPP
