"""Creates an extinction relation using dustapprox for GaiaNIR photometric bands. This
allows for extinction in a given GaiaNIR band (N, R, J, H, or K) to be queried.

N.B.: This script saves intermediate output to
results/_temporary_files/photometric_model.ecsv.

Runtime: 120-240 minutes on your laptop
"""

import numpy as np
from dustapprox.tools import generate_model
from gaianir_open_clusters.config import DATA_DIRECTORY, RESULTS_DIRECTORY
from gaianir_open_clusters.photometry import FILTERS


# Step 1
# Calculate photometry in bins of size 0.5 from 0 to 50 A_V
print("Calculating photometry...")

extinction_values = np.linspace(0.0, 50, num=101)
model_pattern = (DATA_DIRECTORY / "models/Kurucz2003all/*.fl.dat.txt").as_posix()
outdir = RESULTS_DIRECTORY / "_temporary_files"
outdir.mkdir(exist_ok=True, parents=True)

result = generate_model.generate_grid(
    model_pattern,
    outdir / "photometric_model.ecsv",
    list(FILTERS.values()),
    atmosphere_name="Kurucz (ODFNEW/NOVER 2003)",
    extinction_curve="G23",
    A0=extinction_values,
    n_jobs=6,
)

# Step 2
# Fit an extinction model to these data
print("Done! Now fitting A_X/A_0 models...")
extinction_model = RESULTS_DIRECTORY / "extinction_model.ecsv"
features = "teff A0 R0".split()

models = generate_model.train_polynomial_model(
    result, extinction_model, features, degree=3
)
generate_model.export_trained_model_to_ecsv(extinction_model, models)
