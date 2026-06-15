"""Simulates regions of the MW observed by GaiaNIR."""

import numpy as np
import pickle
from gaianir_open_clusters.config import RESULTS_DIRECTORY
from gaianir_open_clusters.population import simulate_region

from pathlib import Path


# SETTINGS
outdir = RESULTS_DIRECTORY / "regions"
outdir.mkdir(exist_ok=True, parents=True)
longitudes = np.linspace(0, 360, num=60, endpoint=False)
latitude = 0
start_area = 1 / 60**2
minimum_stars = 10000

print(f"Total: {len(longitudes)} regions to generate")


# GENERATION SETUP
# Work out what needs doing
done_files = outdir.glob("*.parquet")
done_longitudes = [float(x.stem.split("_")[0]) for x in done_files]
longitudes_to_do = longitudes[np.isin(longitudes, done_longitudes, invert=True)]

print(f"{len(longitudes) - len(longitudes_to_do)} are already done")

# Generate them!
for i, longitude in enumerate(longitudes):
    print(f"Longitude {i+1} of {len(longitudes)} (l={longitude:.2f})")
    region, metadata = simulate_region(
        longitude, latitude, start_area, minimum_stars=minimum_stars
    )
    region.to_parquet(
        outdir
        / f"{longitude:.3f}_{latitude:.3f}.parquet"
    )
    with open(outdir / f"{longitude:.3f}_{latitude:.3f}.parquet", 'wb') as file:
        pickle.dump(metadata, file)
