"""Finite-difference time-domain solution of Maxwell's equations in 1D and 2D (TMz) with a CPML boundary.

Normalised units: speed of light = 1, cell size = 1, impedance of free space = 1.
"""
import numpy as np


def run_1d(eps, steps, source_at, probe_at, pulse):
    """1D FDTD with time step 1 (exact in vacuum) and exact one-way boundaries in the vacuum end cells."""
    n = eps.size
    e, h = np.zeros(n), np.zeros(n - 1)
    record = np.zeros(steps)
    for t in range(steps):
        h += e[1:] - e[:-1]
        left, right = e[1], e[-2]
        e[1:-1] += (h[1:] - h[:-1]) / eps[1:-1]
        e[source_at] += pulse(t)
        e[0], e[-1] = left, right
        record[t] = e[probe_at]
    return record


def slab_transmission(eps_r=4.0, thickness=40, steps=8192, scale=1):
    """Power transmission of a dielectric slab: (frequencies, simulated, analytic)."""
    n, start = 1200, 500
    thickness, steps = thickness * scale, steps * scale          # finer grid, same physical problem
    vacuum = np.ones(n)
    slab = vacuum.copy()
    slab[start:start + thickness + 1] = eps_r
    slab[[start, start + thickness]] = (1 + eps_r) / 2          # interface nodes lie half in each medium

    def pulse(t):
        return np.exp(-((t - 120 * scale) / (20 * scale)) ** 2)

    a = run_1d(vacuum, steps, 200, 900, pulse)
    b = run_1d(slab, steps, 200, 900, pulse)
    f = np.fft.rfftfreq(steps)
    band = (f > 0.002 / scale) & (f < 0.025 / scale)
    simulated = np.abs(np.fft.rfft(b)[band] / np.fft.rfft(a)[band]) ** 2
    index = np.sqrt(eps_r)
    finesse = ((index ** 2 - 1) / (2 * index)) ** 2
    analytic = 1 / (1 + finesse * np.sin(2 * np.pi * f[band] * index * thickness) ** 2)
    return f[band], simulated, analytic


class Grid2D:
    """TMz fields (Ez, Hx, Hy). Outer edges are perfect conductors; `pml` cells in front of them absorb."""

    def __init__(self, nx, ny, courant=0.99 / np.sqrt(2), pml=0, eps=None):
        self.dt = courant
        self.ez = np.zeros((nx, ny))
        self.hx = np.zeros((nx, ny - 1))
        self.hy = np.zeros((nx - 1, ny))
        self.eps = np.ones((nx, ny)) if eps is None else eps
        self.pml = pml
        if pml:
            self.bx_e, self.cx_e = self._profile(np.arange(1, nx - 1), nx)
            self.by_e, self.cy_e = self._profile(np.arange(1, ny - 1), ny)
            self.bx_h, self.cx_h = self._profile(np.arange(nx - 1) + 0.5, nx)
            self.by_h, self.cy_h = self._profile(np.arange(ny - 1) + 0.5, ny)
            self.psi_ezx = np.zeros((nx - 2, ny - 2))
            self.psi_ezy = np.zeros((nx - 2, ny - 2))
            self.psi_hx = np.zeros((nx, ny - 1))
            self.psi_hy = np.zeros((nx - 1, ny))

    def _profile(self, position, n, order=3, alpha_max=0.05):
        """Recursive-convolution coefficients with polynomial conductivity grading."""
        depth = np.maximum(self.pml - position, position - (n - 1 - self.pml))
        depth = np.clip(depth / self.pml, 0, 1)
        sigma = 0.8 * (order + 1) * depth ** order
        alpha = alpha_max * (1 - depth)
        b = np.exp(-(sigma + alpha) * self.dt)
        c = np.where(sigma > 0, sigma / (sigma + alpha + 1e-30) * (b - 1), 0.0)
        return b, c

    def step(self):
        dt, ez = self.dt, self.ez
        dy = ez[:, 1:] - ez[:, :-1]
        dx = ez[1:, :] - ez[:-1, :]
        if self.pml:
            self.psi_hx = self.by_h * self.psi_hx + self.cy_h * dy
            self.psi_hy = self.bx_h[:, None] * self.psi_hy + self.cx_h[:, None] * dx
            dy = dy + self.psi_hx
            dx = dx + self.psi_hy
        self.hx -= dt * dy
        self.hy += dt * dx
        dhy = self.hy[1:, 1:-1] - self.hy[:-1, 1:-1]
        dhx = self.hx[1:-1, 1:] - self.hx[1:-1, :-1]
        if self.pml:
            self.psi_ezx = self.bx_e[:, None] * self.psi_ezx + self.cx_e[:, None] * dhy
            self.psi_ezy = self.by_e * self.psi_ezy + self.cy_e * dhx
            dhy = dhy + self.psi_ezx
            dhx = dhx + self.psi_ezy
        ez[1:-1, 1:-1] += dt / self.eps[1:-1, 1:-1] * (dhy - dhx)


def cavity_resonance(nx, ny, m, n, steps=20000):
    """Excite mode (m, n) of a closed rectangular cavity and measure its frequency."""
    g = Grid2D(nx, ny)
    x, y = np.arange(nx) / (nx - 1), np.arange(ny) / (ny - 1)
    g.ez[:] = np.outer(np.sin(m * np.pi * x), np.sin(n * np.pi * y))
    record = np.zeros(steps)
    probe = (nx // (2 * m), ny // (2 * n))
    for t in range(steps):
        g.step()
        record[t] = g.ez[probe]
    spectrum = np.abs(np.fft.rfft(record * np.hanning(steps), 8 * steps))
    k = int(np.argmax(spectrum))
    a, b, c = np.log(spectrum[k - 1:k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) / (8 * steps * g.dt)


def analytic_resonance(nx, ny, m, n):
    return 0.5 * np.hypot(m / (nx - 1), n / (ny - 1))


def discrete_resonance(nx, ny, m, n, courant=0.99 / np.sqrt(2)):
    """Exact frequency of the finite-difference scheme (includes numerical dispersion)."""
    s = np.hypot(np.sin(m * np.pi / (2 * (nx - 1))), np.sin(n * np.pi / (2 * (ny - 1))))
    return np.arcsin(courant * s) / (np.pi * courant)


def pml_reflection(pml, size=80, steps=300):
    """Largest error (dB, relative to the signal peak) compared with a domain too large to reflect."""
    big = size + 2 * steps
    small, reference = Grid2D(size, size, pml=pml), Grid2D(big, big)
    offset = (big - size) // 2
    probe = (pml + 3 if pml else 3, size // 2)
    error = peak = 0.0
    for t in range(steps):
        value = np.exp(-((t - 40) / 12) ** 2) * (-(t - 40) / 12)
        small.step()
        reference.step()
        small.ez[size // 2, size // 2] += value
        reference.ez[offset + size // 2, offset + size // 2] += value
        r = reference.ez[offset + probe[0], offset + probe[1]]
        error = max(error, abs(small.ez[probe] - r))
        peak = max(peak, abs(r))
    return 20 * np.log10(error / peak)
