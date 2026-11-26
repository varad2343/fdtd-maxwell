# FDTD Electromagnetic Solver

**Subject: Electromagnetic Engineering**

Maxwell's equations in 1D and 2D with an absorbing boundary, checked against analytical results.

## Results
- Dielectric slab error converges at second order
- Cavity resonances match the discrete dispersion relation
- 10-cell absorbing layer: about -85 dB peak error in the tested set-up

User guide: [docs/User_Guide.pdf](docs/User_Guide.pdf).

## Run
```bash
pip install -r requirements.txt
python -m pytest
python experiments.py
```

## Limitations
Lossless, non-dispersive media; 1D and 2D.
