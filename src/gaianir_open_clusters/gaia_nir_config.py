"""File containing various defaults for GaiaNIR. These can be changed to alter the
parameters of the simulation. This file, and this file ONLY, contains hyperparameters
used by the entire project.
"""

import pooch
from pathlib import Path
from gaianir_open_clusters.config import DATA_DIRECTORY


# ASTROMETRY
# By default, we use a file provided by David Hobbs based on simulations from ~January
# 2026 for this project. The simulations are divided by mission type and spectral class.
# We smoothly interpolate between stars of different type to fill in the gaps here.
# Path to the file used
ASTROMETRIC_DATA = DATA_DIRECTORY / "EoM_Astrometry_2026_January"

# Its location online
ASTROMETRIC_DATA_URL = "https://cloud.emily.space/public.php/dav/files/Z3jmFLLrxHsXkJM"

# Minimum observable separation between stars that GaiaNIR can handle before they are resolved as the same source
GAIA_NIR_MINIMUM_SEPARATION = 0.6


# PHOTOMETRY
# Faintest N-band magnitude that the telescope will observe
FAINTEST_N_MAGNITUDE = 24

# Parameters of the various photometric bands. Right now, they are just tophat functions
# with an assumed 70% transmissivity.
GAIA_NIR_FILTERS = {
    "N": {"mid": 1550.0, "width": 1500, "transmission": 0.7},
    "N_R": {"mid": 975, "width": 350, "transmission": 0.7},
    "N_J": {"mid": 1275, "width": 250, "transmission": 0.7},
    "N_H": {"mid": 1600, "width": 400, "transmission": 0.7},
    "N_K": {"mid": 2050, "width": 500, "transmission": 0.7},
}
