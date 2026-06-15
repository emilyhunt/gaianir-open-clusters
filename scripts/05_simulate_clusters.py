import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

from gaianir_open_clusters.util import get_circular_orbit_skycoord


longitudes = np.linspace(0, 360, num=60, endpoint=False)
latitude = 0.0
distances = np.linspace(2000, 20000, num=10)

parameters = dict(
    pleiades=dict(
        mass=1000,
        log_age=9,
        metallicity=0.0,
        r_core=2.3,
        r_tidal=11,
        virial_ratio=0.5,
    ),
    embedded=dict(
        mass=1000,
        log_age=6,
        metallicity=0.0,
        r_core=0.5,
        r_tidal=3,
        virial_ratio=0.5,
    ),
    # big_embedded=dict(
    #     mass=25000,
    #     log_age=6.0,
    #     metallicity=0.0,
    #     r_core=0.75,
    #     r_tidal=5,
    #     virial_ratio=0.5,
    # ),
    berkeley_29=dict(
        mass=1500,
        log_age=9.5,
        metallicity=-0.5,
        r_core=2.5,
        r_tidal=15,
        virial_ratio=0.5,
    ),
    # globular=dict(
    #     mass=20000,
    #     log_age=10,
    #     metallicity=-1.5,
    #     r_core=2.5,
    #     r_tidal=15,
    #     virial_ratio=0.5,
    # ),
)


def get_params_and_models(cluster, l, b, distance):
    params = SimulatedClusterParameters(
        position=get_circular_orbit_skycoord(l, b, distance), **parameters[cluster]
    )
    models = SimulatedClusterModels(
        observations=[
            GaiaNIRObservationModel(mission_class="GaiaNIR-L", years=10),
            GaiaNIRObservationModel(mission_class="GaiaNIR-M", years=10),
        ]
    )
    return params, models


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


for cluster, params in parameters.items():
    outdir = RESULTS_DIRECTORY / f"clusters/{cluster}"
    outdir.mkdir(exist_ok=True, parents=True)
    for longitude in longitudes:
        for distance in distances:
            print(
                f"\rcluster {cluster}, l={longitude:.2f}, d={distance:.2f}",
                end=" " * 10,
            )
            outfile = (
                outdir
                / f"{cluster}_{longitude:.3f}_{latitude:.3f}_{distance:.3f}.parquet"
            )
            if outfile.exists():
                continue

            params, models = get_params_and_models(
                cluster, longitude, latitude, distance
            )
            simulated_cluster = SimulatedCluster(parameters=params, models=models)
            simulated_cluster.make()
            # todo more here, including adding crowding, removing unnecessary columns, and combining into one dataframe
