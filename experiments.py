"""Reproduces the reported numbers and figures in results/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtd import analytic_resonance, cavity_resonance, discrete_resonance, pml_reflection, slab_transmission

out = Path("results")
out.mkdir(exist_ok=True)
f, sim, exact = slab_transmission()
cavity = [{"mode": f"({m},{n})", "analytic": analytic_resonance(61, 41, m, n),
           "discrete": discrete_resonance(61, 41, m, n), "measured": cavity_resonance(61, 41, m, n)}
          for m, n in ((1, 1), (2, 1), (1, 2), (3, 2), (1, 3))]
pml = {str(d): pml_reflection(d) for d in (0, 4, 6, 8, 10, 14, 20)}
slab = {str(s): float(np.max(np.abs(np.subtract(*slab_transmission(scale=s)[1:])))) for s in (1, 2, 4)}
results = {"slab_max_error": slab, "cavity": cavity, "pml_db": pml}
print(json.dumps(results, indent=1))
(out / "results.json").write_text(json.dumps(results, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(f, exact, label="analytic")
ax[0].plot(f, sim, "k:", label="FDTD")
ax[0].set(title="Dielectric slab, $\\epsilon_r$ = 4", xlabel="Frequency (c / cell)", ylabel="Power transmission")
ax[0].legend()
ax[1].plot([int(k) for k in pml], list(pml.values()), "o-")
ax[1].set(title="CPML reflection", xlabel="Layer thickness (cells)", ylabel="Peak error (dB)")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "fdtd.png", dpi=160)
