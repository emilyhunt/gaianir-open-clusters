"""Simulates regions of the MW observed by GaiaNIR."""

import numpy as np
import pickle
from gaianir_open_clusters.config import RESULTS_DIRECTORY
from gaianir_open_clusters.gaia_nir_config import (
    SIMULATION_LONGITUDES,
    SIMULATION_LATITUDE,
)
from gaianir_open_clusters.population import simulate_region


# SETTINGS
outdir = RESULTS_DIRECTORY / "regions"
outdir.mkdir(exist_ok=True, parents=True)
start_area = 1 / 60**2
minimum_stars = 10000

print(f"Total: {len(SIMULATION_LONGITUDES)} regions to generate")


# GENERATION SETUP
# Work out what needs doing
done_files = outdir.glob("*.parquet")
done_longitudes = [float(x.stem.split("_")[0]) for x in done_files]
longitudes_to_do = SIMULATION_LONGITUDES[
    np.isin(SIMULATION_LONGITUDES, done_longitudes, invert=True)
]

print(f"{len(SIMULATION_LONGITUDES) - len(longitudes_to_do)} are already done")

# Generate them!
for i, longitude in enumerate(longitudes_to_do):
    print(f"Longitude {i + 1} of {len(longitudes_to_do)} (l={longitude:.2f})")
    region, metadata = simulate_region(
        longitude, SIMULATION_LATITUDE, start_area, minimum_stars=minimum_stars
    )
    region.to_parquet(outdir / f"{longitude:.3f}_{SIMULATION_LATITUDE:.3f}.parquet")
    with open(
        outdir / f"{longitude:.3f}_{SIMULATION_LATITUDE:.3f}_metadata.pickle", "wb"
    ) as file:
        pickle.dump(metadata, file)
