"""File containing various defaults for GaiaNIR. These can be changed to alter the
parameters of the simulation. This file, and this file ONLY, contains hyperparameters
used by the entire project.
"""

import pooch
from pathlib import Path
from gaianir_open_clusters.config import DATA_DIRECTORY
from gala.potential import MilkyWayPotential


# ASTROMETRY
# By default, we use a file provided by David Hobbs based on simulations from ~January
# 2026 for this project. The simulations are divided by mission type and spectral class.
# We smoothly interpolate between stars of different type to fill in the gaps here.
# Path to the file used
ASTROMETRIC_DATA = DATA_DIRECTORY / "EoM_Astrometry_2026_January"

# Its location online
ASTROMETRIC_DATA_URL = "https://cloud.emily.space/public.php/dav/files/Z3jmFLLrxHsXkJM"

# For combining astrometry
# Whether or not to do it
COMBINE_GAIA_GAIANIR_ASTROMETRY = True

# Assumed separation between the Gaia end of mission and the GaiaNIR launch date
GAIA_GAIANIR_SEPARATION = 20

# PHOTOMETRY
# Faintest N-band or G-band magnitudes that the telescopes will observe
FAINTEST_N_MAGNITUDE = 24
FAINTEST_GAIA_MAGNITUDE = 21

# Assumed faintest magnitudes for simulations. These aren't quite the faintest possible,
# but replicate common quality cuts - a bit like cutting G<18 with Gaia data.
# GaiaNIR faintest mag for simulating the catalogue and clusters
FAINTEST_N_MAGNITUDE_USED = 23

# Faintest Gaia mag used for combined astrometry
FAINTEST_GAIA_MAGNITUDE_USED = 20

# Parameters of the various photometric bands. Right now, they are just tophat functions
# with an assumed 70% transmissivity.
GAIA_NIR_FILTERS = {
    "N": {"mid": 1550.0, "width": 1500, "transmission": 0.7},
    "N_R": {"mid": 975, "width": 350, "transmission": 0.7},
    "N_J": {"mid": 1275, "width": 250, "transmission": 0.7},
    "N_H": {"mid": 1600, "width": 400, "transmission": 0.7},
    "N_K": {"mid": 2050, "width": 500, "transmission": 0.7},
}

# CROWDING / TELESCOPE APERTURE
# Apertures of GaiaNIR mission designs & Gaia, in metres.
GAIA_NIR_APERTURES = {
    "Gaia": 1.45,
    "GaiaNIR-M": 1.7,
    "GaiaNIR-L": 3.5,
}

# Effective wavelength for source identification / astrometry
GAIA_NIR_EFFECTIVE_WAVELENGTHS = {
    "Gaia": 673,
    "GaiaNIR-M": 1550,
    "GaiaNIR-L": 1550,
}

# Minimum observable separation between stars that GaiaNIR can handle before they are
# resolved as the same source
# N.B.: this is in radians!
GAIANIR_ANGULAR_RESOLUTION = {
    telescope: 1.22 * GAIA_NIR_EFFECTIVE_WAVELENGTHS[telescope] / 1e9 / aperture
    for telescope, aperture in GAIA_NIR_APERTURES.items()
}


# MILKY WAY PARAMETERS
POTENTIAL = MilkyWayPotential(version="v2")
