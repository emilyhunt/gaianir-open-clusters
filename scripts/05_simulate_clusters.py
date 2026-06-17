import numpy as np
import pandas as pd
import pickle
from astropy.coordinates import SkyCoord
from astropy import units as u
from gaianir_open_clusters.config import RESULTS_DIRECTORY
from gaianir_open_clusters.cluster_model import (
    GaiaNIRObservationModel,
    combine_observations,
)
from gaianir_open_clusters.gaia_nir_config import (
    SIMULATION_CLUSTER_PARAMETERS,
    SIMULATION_DISTANCES,
    SIMULATION_LONGITUDES,
    SIMULATION_LATITUDE,
    DIFFERENTIAL_EXTINCTION_FACTOR,
)
from gaianir_open_clusters.util import get_circular_orbit_skycoord
from ocelot.simulate import (
    SimulatedCluster,
    SimulatedClusterParameters,
    SimulatedClusterModels,
)
from dustmaps.bayestar import BayestarQuery
from dustmaps.decaps import DECaPSQueryLite
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import sys


# SETTINGS
# Whether or not to force re-generate pre-existing clusters
# If you change cluster settings, you may need to do this - as they will not
# automatically be re-generated
FORCE_REGENERATION = False
PROCESSES = 6


def get_clusters_to_simulate():
    """Fetches a dataframe containing the parameters of all clusters to simulate."""
    # Grab positions
    longitudes, distances = [
        x.flatten() for x in np.meshgrid(SIMULATION_LONGITUDES, SIMULATION_DISTANCES)
    ]
    position = get_circular_orbit_skycoord(
        longitudes, np.full_like(longitudes, SIMULATION_LATITUDE), distances
    )

    # Grab extinction
    _bayestar_map = BayestarQuery(max_samples=1)
    _zucker_map = DECaPSQueryLite(mean_only=True)

    extinction_bayestar = _bayestar_map.query(position, mode="best")
    extinction_zucker = _zucker_map.query(position, mode="mean")
    extinction = np.where(
        np.isfinite(extinction_zucker), extinction_zucker, extinction_bayestar
    )
    del _bayestar_map, _zucker_map

    # Create a dataframe of everything to simulate
    clusters = []
    for cluster, params in SIMULATION_CLUSTER_PARAMETERS.items():
        extinction_boost = 0.0
        if "extinction_boost" in params:
            extinction_boost = params.pop("extinction_boost")
        extinction_cluster = extinction + extinction_boost

        outdir = f"clusters/{cluster}/"
        paths = [
            outdir + f"{l:.3f}_{SIMULATION_LATITUDE:.3f}_{d:.3f}.parquet"
            for l, d in zip(longitudes, distances)
        ]
        region_paths = [
            "regions/" + f"{l:.3f}_{SIMULATION_LATITUDE:.3f}.parquet"
            for l in longitudes
        ]

        clusters.append(
            pd.DataFrame.from_dict(
                dict(
                    cluster=cluster,
                    l=longitudes,
                    b=SIMULATION_LATITUDE,
                    path=paths,
                    path_region=region_paths,
                    distance=distances,
                    extinction=extinction_cluster,
                    differential_extinction=DIFFERENTIAL_EXTINCTION_FACTOR
                    * extinction_cluster,
                    ra=position.ra.value,
                    dec=position.dec.value,
                    pmra=position.pm_ra_cosdec.value,
                    pmdec=position.pm_dec.value,
                    radial_velocity=position.radial_velocity.value,
                    **params,
                )
            )
        )

    return (
        pd.concat(clusters, ignore_index=True)
        .sort_values(["cluster", "l", "b", "distance"])
        .reset_index(drop=True)
    )


def restrict_to_unsimulated_clusters(clusters):
    # Work out which ones already exist
    clusters["simulated"] = False
    if not FORCE_REGENERATION:
        for i_row, a_row in clusters.iterrows():
            if (RESULTS_DIRECTORY / a_row["path"]).exists():
                clusters.loc[i_row, "simulated"] = True

    to_simulate = clusters.query("not simulated").reset_index(drop=True)
    clusters = clusters.drop(columns=["simulated"])
    return to_simulate


def get_params_and_models(params):
    position = SkyCoord(
        ra=params["ra"] * u.deg,
        dec=params["dec"] * u.deg,
        pm_ra_cosdec=params["pmra"] * u.mas / u.yr,
        pm_dec=params["pmdec"] * u.mas / u.yr,
        distance=params["distance"] * u.pc,
        radial_velocity=params["radial_velocity"] * u.km / u.s,
    )

    params = SimulatedClusterParameters(
        position=position,
        mass=params["mass"],
        log_age=params["log_age"],
        metallicity=params["metallicity"],
        r_core=params["r_core"],
        r_tidal=params["r_tidal"],
        extinction=params["extinction"],
        differential_extinction=params["differential_extinction"],
        virial_ratio=params["virial_ratio"],
    )
    models = SimulatedClusterModels(
        observations=[
            GaiaNIRObservationModel(mission_class="GaiaNIR-L", years=10),
            GaiaNIRObservationModel(mission_class="GaiaNIR-M", years=10),
            GaiaNIRObservationModel(
                mission_class="Gaia",
                years=10,
                maximum_magnitude=21,
                combined_astrometry=False,
            ),
            GaiaNIRObservationModel(
                mission_class="Gaia",
                years=5,
                maximum_magnitude=21,
                combined_astrometry=False,
            ),
        ]
    )
    return params, models


def simulate_cluster(df_row: pd.Series):
    params, models = get_params_and_models(df_row)

    metadata_file = (
        RESULTS_DIRECTORY
        / f"regions/{df_row['l']:.3f}_{df_row['b']:.3f}_metadata.pickle"
    )

    with open(metadata_file, "rb") as file:
        crowding_metadata = pickle.load(file)

    cluster = SimulatedCluster(parameters=params, models=models)
    cluster.make()
    observations = combine_observations(cluster.observations, crowding_metadata)

    outfile = Path(RESULTS_DIRECTORY / df_row["path"])
    outfile.parent.mkdir(exist_ok=True, parents=True)
    observations.to_parquet(outfile)
    
    sys.stdout.flush()


if __name__ == "__main__":
    print("Setting up list of clusters to simulate...")
    clusters = get_clusters_to_simulate()
    clusters.to_parquet(RESULTS_DIRECTORY / "simulated_clusters.parquet")
    print(f"Generated {len(clusters)} to simulate")
    to_simulate = restrict_to_unsimulated_clusters(clusters)
    print(f"... of which {len(to_simulate)} require simulating.")

    # god i hate pandas (this is an easy way to get single rows as an iterable)
    rows_as_list = [a_row for i_row, a_row in to_simulate.iterrows()]

    map = process_map(
        simulate_cluster,
        rows_as_list,
        max_workers=PROCESSES,
        chunksize=1,
        # miniters=1,
        # maxinterval=1,
    )
