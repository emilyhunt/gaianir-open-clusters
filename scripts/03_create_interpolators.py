"""Creates interpolator assets for use generating photometry.

Runtime: <1 minute
"""

import pandas as pd
from gaianir_open_clusters.config import RESULTS_DIRECTORY
from gaianir_open_clusters.photometry import FILTERS


# Grab raw photometric results
print("Loading data...")
kurucz_photometry = (
    pd.read_csv(
        RESULTS_DIRECTORY / "_temporary_files/photometric_model.ecsv", comment="#"
    )
    .query("A0==0.0 and alpha==0.0 and R0==3.1")
    .reset_index(drop=True)
)
btsettl_photometry = pd.read_csv(
    RESULTS_DIRECTORY / "_temporary_files/btsettl_photometric_model.ecsv", comment="#"
)

# Process them into a nicer format, with each band as a column (instead of many rows)
PHOTOMETRIC_GRID = pd.concat([kurucz_photometry, btsettl_photometry])

bands = PHOTOMETRIC_GRID["passband"].unique()

for band in bands:
    filter = FILTERS[band]

    # Copy photometry just to the valid set
    valid = PHOTOMETRIC_GRID["passband"] == band
    PHOTOMETRIC_GRID.loc[valid, band] = (
        PHOTOMETRIC_GRID.loc[valid, "mag0"] - filter.Vega_zero_mag
    )

mags = {band: pd.NamedAgg(band, "first") for band in bands}

PHOTOMETRIC_GRID = (
    PHOTOMETRIC_GRID.groupby(["teff", "logg", "feh"]).aggregate(**mags).reset_index()
)
PHOTOMETRIC_GRID.to_parquet(RESULTS_DIRECTORY / "photometry.parquet")
