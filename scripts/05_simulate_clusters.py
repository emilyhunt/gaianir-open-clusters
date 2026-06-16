import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia

# import gaianir_open_clusters  # noqa: F401
from gaianir_open_clusters.config import RESULTS_DIRECTORY
from gaianir_open_clusters.cluster_model import GaiaNIRObservationModel
from gaianir_open_clusters.population import simulate_region
from gaianir_open_clusters.gaia_nir_config import (
    FAINTEST_N_MAGNITUDE_USED,
    FAINTEST_GAIA_MAGNITUDE_USED,
    SIMULATION_CLUSTER_PARAMETERS,
    SIMULATION_DISTANCES,
    SIMULATION_LONGITUDES,
    SIMULATION_LATITUDE,
    DIFFERENTIAL_EXTINCTION_FACTOR,
)
from gaianir_open_clusters.crowding import apply_cluster_crowding
from ocelot.simulate import (
    SimulatedCluster,
    SimulatedClusterParameters,
    SimulatedClusterModels,
)
from ocelot.model.observation import (
    GaiaDR3ObservationModel,
    GenericSubsampleSelectionFunction,
)
from scipy.stats import poisson, ecdf
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from dustmaps.bayestar import BayestarQuery
from dustmaps.decaps import DECaPSQueryLite

from gaianir_open_clusters.util import get_circular_orbit_skycoord


# SETTINGS
# Whether or not to force re-generate pre-existing clusters
# If you change cluster settings, you may need to do this - as they will not
# automatically be re-generated
FORCE_REGENERATION = False

unnecessary_columns = [
    "simulated_id",
    "cluster_id",
    "simulated_star",
    "gaia_dr3_g_true",
    "gaia_dr3_bp_true",
    "gaia_dr3_rp_true",
    "companions",
    "mass_ratio",
    "period",
    "eccentricity",
    "simulated_id_primary",
    "ra",
    "dec",
    "pmra_true",
    "pmdec_true",
    "parallax_true",
    "radial_velocity_true",
    "extinction",
    "gaianir_n_true",
    "gaianir_r_true",
    "gaianir_j_true",
    "gaianir_h_true",
    "gaianir_k_true",
    "extinction_gaianir_n",
    "extinction_gaianir_r",
    "extinction_gaianir_j",
    "extinction_gaianir_h",
    "extinction_gaianir_k",
    "unresolved_companions",
    "selection_probability_GaiaNIRSelectionFunction",
    "selection_probability",
]


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

        clusters.append(
            pd.DataFrame.from_dict(
                dict(
                    cluster=cluster,
                    l=longitudes,
                    b=SIMULATION_LATITUDE,
                    path=paths,
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
            # todo add missing Gaia sim
        ]
    )
    return params, models


if __name__ == "__main__":
    print("Setting up list of clusters to simulate...")
    clusters = get_clusters_to_simulate()
    clusters.to_parquet(RESULTS_DIRECTORY / "simulated_clusters.parquet")
    print(f"Generated {len(clusters)} to simulate")
    to_simulate = restrict_to_unsimulated_clusters(clusters)
    print(f"... of which {len(to_simulate)} require simulating.")
