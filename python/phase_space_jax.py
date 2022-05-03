import math
from typing import Callable, Optional

import chex
import jax.numpy as jnp
import numpy as np

import jax

SquaredMatrixElement = Callable[[chex.Array], chex.Numeric]


def kallen_lambda(a, b, c):
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * a * c - 2 * b * c


def _rambo_init(key: chex.Array, n: int, batch_size: int) -> chex.Array:
    keys = jax.random.split(key, 4)
    rho1 = jax.random.uniform(keys[0], shape=(n, batch_size))
    rho2 = jax.random.uniform(keys[1], shape=(n, batch_size))
    rho3 = jax.random.uniform(keys[2], shape=(n, batch_size))
    rho4 = jax.random.uniform(keys[3], shape=(n, batch_size))

    ctheta = 2 * rho1 - 1.0
    stheta = jnp.sqrt(1.0 - ctheta**2)
    phi = 2.0 * jnp.pi * rho2
    e = -jnp.log(rho3 * rho4)

    return jnp.array(
        [e, e * stheta * jnp.cos(phi), e * stheta * jnp.sin(phi), e * ctheta]
    )


def _rambo_boost(momenta: chex.Array, cme: float):
    # expecting an array with dimension (4, # final-state-particles)
    # chex.assert_rank(momenta, 2)

    sum_ps = jnp.sum(momenta, axis=1)
    inv_mass = jnp.sqrt(
        sum_ps[0] ** 2 - sum_ps[1] ** 2 - sum_ps[2] ** 2 - sum_ps[3] ** 2
    )
    inv_mass = 1.0 / inv_mass

    bx = -inv_mass * sum_ps[1]
    by = -inv_mass * sum_ps[2]
    bz = -inv_mass * sum_ps[3]

    x = cme * inv_mass
    g = sum_ps[0] * inv_mass
    a = 1.0 / (1.0 + g)

    bdotp = bx * momenta[1] + by * momenta[2] + bz * momenta[3]
    fact = a * bdotp + momenta[0]

    return jnp.array(
        [
            x * (g * momenta[0] + bdotp),
            x * (fact * bx + momenta[1]),
            x * (fact * by + momenta[2]),
            x * (fact * bz + momenta[3]),
        ]
    )


def _rambo_compute_scale_factor(
    momenta: chex.Array, cme: float, masses: chex.Array, iterations: int
) -> chex.Numeric:
    e = momenta[0]

    xi0 = jnp.sqrt(1.0 - jnp.square(jnp.sum(masses) / cme))

    xi = xi0 * jnp.ones_like(e[0])
    for _ in range(iterations):
        deltaf = jnp.hypot(e * xi, masses)
        f = jnp.sum(deltaf, axis=0) - cme
        df = jnp.sum(xi * jnp.square(e) / deltaf, axis=0)
        xi = xi - f / df

    return xi


def _rambo_correct_masses(
    momenta: chex.Array, cme: float, masses: chex.Array, iterations: int
) -> chex.Array:
    xi = _rambo_compute_scale_factor(momenta, cme, masses, iterations)
    return jnp.array(
        [
            jnp.hypot(xi * momenta[0], masses),
            xi * momenta[1],
            xi * momenta[2],
            xi * momenta[3],
        ]
    )


def _rambo_weight_rescale_factors(
    momenta: chex.Array, cme: float, n: int
) -> chex.Numeric:
    modsqr = jnp.sum(jnp.square(momenta[1:]), axis=0)
    mod = jnp.sqrt(modsqr)

    t1 = jnp.sum(mod / cme, axis=0) ** (2 * n - 3)
    t2 = jnp.sum(modsqr / momenta[0], axis=0)
    t3 = jnp.prod(mod / momenta[0], axis=0)

    return t1 / t2 * t3 * cme


def _rambo_compute_weights(momenta: chex.Array, cme: float, n: int, base_wgt: float):
    return _rambo_weight_rescale_factors(momenta, cme, n) * base_wgt


def _rambo_base_weight(n: int, cme: float):
    pi = np.pi
    fact_nm2 = math.factorial(n - 2)
    fact = 1.0 / ((n - 1) * fact_nm2**2)
    return fact * (0.5 * pi) ** (n - 1) * cme ** (2 * n - 4) * (0.5 / pi) ** (3 * n - 4)


def _rambo_generate(
    cme: float,
    masses: chex.Array,
    key: chex.Array,
    batch_size: int,
    iterations: int,
    base_wgt: float,
):
    n = len(masses)
    momenta = _rambo_init(key, n, batch_size)
    momenta = _rambo_boost(momenta, cme)
    momenta = _rambo_correct_masses(momenta, cme, masses, iterations)
    weights = _rambo_compute_weights(momenta, cme, n, base_wgt)
    return momenta, weights


def generate_phase_space(
    cme: float,
    masses: np.ndarray,
    key: chex.Array,
    *,
    msqrd: Optional[SquaredMatrixElement] = None,
    batch_size: int = 10_000,
    jit=True,
):
    n: int = len(masses)
    base_wgt = _rambo_base_weight(n, cme)
    ms = jnp.expand_dims(jnp.array(masses), -1)
    iterations = 10

    if jit:
        generator = jax.jit(_rambo_generate, static_argnums=(3, 4, 5, 6))
    else:
        generator = _rambo_generate

    momenta, weights = generator(cme, ms, key, batch_size, iterations, base_wgt)

    if msqrd is not None:
        weights = msqrd(momenta) * weights

    return momenta, weights


def integrate_phase_space(
    cme: float,
    masses: np.ndarray,
    key: chex.Array,
    *,
    msqrd: Optional[SquaredMatrixElement] = None,
    batch_size: int = 10_000,
    jit=True,
):
    assert cme > sum(
        masses
    ), "Center of mass energy must be greater than sum of `masses`."

    _, weights = generate_phase_space(
        cme, masses, key, msqrd=msqrd, batch_size=batch_size, jit=jit
    )

    integral = jnp.nanmean(weights)
    error = jnp.nanstd(weights, ddof=1) / math.sqrt(batch_size)

    return integral, error


def decay_width(
    mass: float,
    masses: np.ndarray,
    key: chex.Array,
    *,
    msqrd: Optional[SquaredMatrixElement] = None,
    batch_size: int = 10_000,
    jit=True,
):
    assert mass > sum(
        masses
    ), "Center of mass energy must be greater than sum of `masses`."

    integral, error = integrate_phase_space(
        mass, masses, key, msqrd=msqrd, batch_size=batch_size, jit=jit
    )

    pre = 0.5 / mass
    width = pre * integral
    error = pre * error

    return width, error


def cross_section(
    cme: float,
    m1: float,
    m2: float,
    masses: np.ndarray,
    key: chex.Array,
    *,
    msqrd: Optional[SquaredMatrixElement] = None,
    batch_size: int = 10_000,
    jit=True,
):
    assert (
        cme > m1 + m2
    ), "Center of mass energy must be greater than sum of `m1` and `m2`."
    assert cme > sum(
        masses
    ), "Center of mass energy must be greater than sum of `masses`."

    integral, error = integrate_phase_space(
        cme, masses, key, msqrd=msqrd, batch_size=batch_size, jit=jit
    )

    pre = 0.5 / math.sqrt(kallen_lambda(cme**2, m1**2, m2**2))
    width = pre * integral
    error = pre * error

    return width, error
