"""Calculates expected photometry for stars observed by GaiaNIR. This mostly uses 
results from 01_extinction_relation.py, but also supplements the Kurucz+03 atmosphere
models with computations done with BT-Settl for low-mass stars.

N.B.: This script saves intermediate output to
results/photometric_model.parquet.

Runtime: 1 minute on your laptop
"""

import numpy as np
from dustapprox.tools import generate_model
from gaianir_open_clusters.config import DATA_DIRECTORY, RESULTS_DIRECTORY
from gaianir_open_clusters.photometry import FILTERS


# Step 1
# Calculate photometry for the additional low-mass stars missed by Kurucz atmospheres
print("Calculating photometry...")

model_pattern = (DATA_DIRECTORY / "models/BT-Settl_LowT/*.dat.txt").as_posix()
outdir = RESULTS_DIRECTORY / "_temporary_files"
outdir.mkdir(exist_ok=True, parents=True)

result = generate_model.generate_grid(
    model_pattern,
    outdir / "btsettl_photometric_model.ecsv",
    list(FILTERS.values()),
    atmosphere_name="BT-Settl",
    extinction_curve="G23",
    A0=np.asarray([0.0]),
    R0=np.asarray([3.1]),
    n_jobs=6,
)
