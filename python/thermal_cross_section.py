from hazma import rambo
from hazma.field_theory_helper_functions.common_functions import minkowski_dot
from scipy import integrate
from scipy import special
import numpy as np


class Model:
    def __init__(self, mx: float) -> None:
        self.mx = mx

    def msqrd(self, momenta) -> float:
        # p1, p2 are DM momenta
        p = sum(momenta)
        q = p[0]  # Center of mass energy
        pi_mag = np.sqrt(q ** 2 / 4.0 - self.mx ** 2)
        p1 = np.array([q / 2.0, 0.0, 0.0, pi_mag])
        p2 = np.array([q / 2.0, 0.0, 0.0, -pi_mag])

        # use `minkowski_dot` to compute scalar products
        # like: minkowski_dot(p1, p2)

        # Final state momenta:
        k1 = momenta[0]
        k2 = momenta[1]
        k3 = momenta[2]
        k4 = momenta[3]

        # Compute cross section ...

        return 1.0

    def cross_section(self, cme: float):
        """
        Compute the zero-temperature cross section.
        """
        fsp_masses = [1.0, 2.0, 3.0, 4.0]
        return rambo.compute_decay_width(
            fsp_masses, cme, mat_elem_sqrd=lambda k: self.msqrd(k)
        )[0]


def thermal_cross_section_integrand(z: float, x: float, model) -> float:
    """
    Compute the integrand of the thermally average cross section for the dark
    matter particle of the given model.
    Parameters
    ----------
    z: float
        Center of mass energy divided by DM mass.
    x: float
        Mass of the dark matter divided by its temperature.
    model: dark matter model
        Dark matter model, i.e. `ScalarMediator`, `VectorMediator`
        or any model with a dark matter particle.
    Returns
    -------
    integrand: float
        Integrand of the thermally-averaged cross-section.
    """
    sig = model.cross_section(model.mx * z)
    kernal = z ** 2 * (z ** 2 - 4.0) * special.k1(x * z)
    return sig * kernal


def thermal_cross_section(x: float, model) -> float:
    """
    Compute the thermally average cross section for the dark
    matter particle of the given model.
    Parameters
    ----------
    x: float
        Mass of the dark matter divided by its temperature.
    model: dark matter model
        Dark matter model, i.e. `ScalarMediator`, `VectorMediator`
        or any model with a dark matter particle.
    Returns
    -------
    tcs: float
        Thermally average cross section.
    """
    # If x is really large, we will get divide by zero errors
    if x > 300:
        return 0.0

    pf = x / (2.0 * special.kn(2, x)) ** 2

    return pf * integrate.quad(
        thermal_cross_section_integrand,
        2.0,
        50.0 / x,
        args=(x, model),
        points=[2.0],
    )[0]


if __name__ == "__main__":
    model = Model(100.0)

    print(thermal_cross_section(1.0, model))
