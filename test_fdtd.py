import numpy as np
import pytest

from fdtd import (Grid2D, analytic_resonance, cavity_resonance, discrete_resonance, pml_reflection, run_1d,
                  slab_transmission)


def test_1d_boundaries_absorb_exactly_in_vacuum():
    record = run_1d(np.ones(300), 1500, 100, 150, lambda t: np.exp(-((t - 100) / 15) ** 2))
    assert np.abs(record[800:]).max() < 1e-10


def test_slab_transmission_matches_fabry_perot_formula():
    errors = []
    for scale in (1, 2):
        _, simulated, analytic = slab_transmission(scale=scale)
        errors.append(np.max(np.abs(simulated - analytic)))
    assert errors[0] < 0.02
    assert errors[0] / errors[1] == pytest.approx(4, rel=0.15)      # second-order convergence


@pytest.mark.parametrize("m,n", [(1, 1), (2, 1), (1, 3), (3, 2)])
def test_cavity_modes(m, n):
    measured = cavity_resonance(61, 41, m, n)
    assert measured == pytest.approx(discrete_resonance(61, 41, m, n), rel=2e-4)
    assert measured == pytest.approx(analytic_resonance(61, 41, m, n), rel=5e-3)


def test_dispersion_error_is_second_order():
    errors = [abs(discrete_resonance(n + 1, n + 1, 2, 1) / analytic_resonance(n + 1, n + 1, 2, 1) - 1)
              for n in (20, 40, 80)]
    assert errors[0] / errors[1] == pytest.approx(4, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(4, rel=0.05)


def test_courant_limit():
    def energy(courant):
        g = Grid2D(40, 40, courant=courant)
        g.ez[20, 20] = 1
        for _ in range(300):
            g.step()
        return np.abs(g.ez).max()

    assert energy(0.70) < 2
    assert energy(0.75) > 1e3


def test_pml_absorbs():
    assert pml_reflection(0) > -20
    assert pml_reflection(10) < -50
