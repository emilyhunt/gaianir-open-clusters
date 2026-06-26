"""Calculates the final detectability of given clusters, based on a little bit of
estimation!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from gaianir_open_clusters.config import (
    RESULTS_DIRECTORY,
)
from gaianir_open_clusters.gaia_nir_config import SIMULATION_FINAL_MAGNITUDE_LIMITS
from gaianir_open_clusters.util import position_to_max_galaxy_distance

from hr_selection_function import (
    HR24SelectionFunction,
    NStarsPredictor,
    GaiaDensityEstimator,
)

from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, multivariate_normal

from astropy.coordinates import SkyCoord
from astropy import units as u


def measure_cluster_density(
    cluster_info,
    members,
    use_combined=True,
    missions=("gaianir-l", "gaianir-m", "gaia_dr5", "gaia_dr4", "gaia_dr3"),
):
    # Add DR3 (DR4 but rescaled)
    members = add_gaia_dr3_uncertainties_based_on_sampling(members)
    members["gaia_dr3"] = members["gaia_dr4"]

    # Cycle over every mission, calculating stuffs
    result = dict()
    for mission in missions:
        maglim = SIMULATION_FINAL_MAGNITUDE_LIMITS[mission]

        subsample = members.loc[
            (members[mission]) & (members[maglim["band"]] < maglim["limit"])
        ]
        key_astrometry = mission
        if use_combined and "gaianir" in mission:
            key_astrometry = mission + "_combined"

        result[f"n_stars_{mission}"] = len(subsample)
        if len(subsample) > 5:
            (
                result[f"rho_position_{mission}"],
                result[f"rho_pm_{mission}"],
                result[f"rho_parallax_{mission}"],
                result[f"rho_astrometric_{mission}"],
                result[f"rho_full_{mission}"],
            ) = _calculate_cluster_density(
                subsample["l"],
                subsample["b"],
                subsample[f"pmra_{key_astrometry}"],
                subsample[f"pmdec_{key_astrometry}"],
                subsample[f"parallax_{key_astrometry}"],
                cluster_info,
            )
        else:
            (
                result[f"rho_position_{mission}"],
                result[f"rho_pm_{mission}"],
                result[f"rho_parallax_{mission}"],
                result[f"rho_astrometric_{mission}"],
                result[f"rho_full_{mission}"],
            ) = 0.0, 0.0, 0.0, 0.0, 0.0

    return result


def _calculate_cluster_density(
    l,
    b,
    pmra,
    pmdec,
    parallax,
    cluster_info,
):
    """Performs successive KDEs to estimate the peak density of a cluster."""
    # Todo this code fails at l=0 - probably not an issue, but bear in mind
    if np.any(l > 270) and np.any(l < 90):
        raise ValueError("Coordinate discontinuities are not supported.")

    n_points = len(l)

    positions = np.vstack((l, b))
    proper_motions = np.vstack((pmra, pmdec))

    kde_pos = gaussian_kde(positions)
    kde_pm = gaussian_kde(proper_motions)
    kde_parallax = gaussian_kde(parallax)
    kde_astro = gaussian_kde(np.vstack((pmra, pmdec, parallax)))
    kde_full = gaussian_kde(np.vstack((l, b, pmra, pmdec, parallax)))

    return (
        n_points * kde_pos((cluster_info["l"], cluster_info["b"]))[0],
        n_points * kde_pm((cluster_info["pmra"], cluster_info["pmdec"]))[0],
        n_points * kde_parallax(cluster_info["parallax"])[0],
        n_points
        * kde_astro(
            (cluster_info["pmra"], cluster_info["pmdec"], cluster_info["parallax"])
        )[0],
        n_points
        * kde_full(
            (
                cluster_info["l"],
                cluster_info["b"],
                cluster_info["pmra"],
                cluster_info["pmdec"],
                cluster_info["parallax"],
            )
        )[0],
    )


def measure_region_density(
    cluster_locations,
    region,
    region_area,
    stars_in_sample_desired: int = 50,
    stars_in_sample_minimum: int = 5,
    stars_in_sample_region_scale: float | int = 3,
    parallax_move_amount=0.1,
    missions=("gaianir-l", "gaianir-m", "gaia_dr5", "gaia_dr4", "gaia_dr3"),
    use_combined: bool = True,
    bandwidth_method: str = "scott",
):
    for mission in missions:
        maglim = SIMULATION_FINAL_MAGNITUDE_LIMITS[mission]
        key_out = mission
        key_proper_motion = mission
        key_parallax = mission

        if use_combined and "gaianir" in mission:
            key_proper_motion = key_proper_motion + "_combined"

        key_crowding = f"uncrowded_{mission.replace('_dr4', '').replace('_dr5', '').replace('_dr3', '')}"  # todo this is ugly af
        region_valid = region.loc[
            (region[key_crowding])
            & (
                region[maglim["band"].replace("g_effective_gaia", "gaia_dr3_g")]
                < maglim["limit"]
            )
        ]
        # if len(region_valid) < minimum_stars_in_sample:
        #     raise ValueError(
        #         f"valid region contains too few stars {len(region_valid)} - consider "
        #         "increasing minimum_stars_in_sample"
        #     )

        this_minimum_stars = np.clip(
            len(region_valid) / stars_in_sample_region_scale,
            stars_in_sample_minimum,
            stars_in_sample_desired,
        )

        # Iterate over each parallax/cluster pm and identify its data density
        for i_row, a_row in cluster_locations.iterrows():
            # Get a sample of stars
            parallax_var = parallax_move_amount
            subsample = []
            while len(subsample) < this_minimum_stars:
                subsample = region_valid.loc[
                    (
                        region_valid[f"parallax_{key_parallax}"]
                        > a_row["parallax"] - parallax_var
                    )
                    & (
                        region_valid[f"parallax_{key_parallax}"]
                        < a_row["parallax"] + parallax_var
                    )
                ]
                parallax_var *= 2

            # Do a KDE to estimate the density at this point
            kde = gaussian_kde(
                subsample[[f"pmra_{key_proper_motion}", f"pmdec_{key_proper_motion}"]]
                .to_numpy()
                .T,
                bw_method=bandwidth_method,
            )
            background_density = kde([a_row["pmra"], a_row["pmdec"]])

            # Do a multivariate normal to estimate the density at this point
            # X = subsample[[f"pmra_{key_proper_motion}", f"pmdec_{key_proper_motion}"]]
            # mean = np.mean(X, axis=0)
            # cov = np.cov(X.T)
            # single_multivariate = multivariate_normal(mean, cov)
            # background_density = single_multivariate.pdf(
            #     [a_row["pmra"], a_row["pmdec"]]
            # )

            # Scale everything based on scale factors so that rho_data
            # is correct - i.e., n_stars mas^{-3} yr^{2} field^{-1}
            width = parallax_var * 2
            cluster_locations.loc[i_row, f"rho_data_pm_{key_out}"] = (
                len(subsample)
                * background_density
                / width  # Parallax width
                / region_area  # Area of underlying simulated region
                # * 30.21  # HR23 clustering field area
            )

            cluster_locations.loc[i_row, f"rho_data_position_{key_out}"] = (
                len(region_valid) / region_area
            )

            # Also record the number of stars, just out of curiosity
            cluster_locations.loc[i_row, f"n_stars_for_rho_data_{key_out}"] = len(
                subsample
            )

    return cluster_locations


def add_gaia_dr3_uncertainties_based_on_sampling(region, seed=None):
    """Adds approximate Gaia DR3 observations to a region, based on back-extrapolating
    uncertainties from Gaia DR4.
    """
    # Make errors based on scale factors
    parallax_scale = np.sqrt(66 / 34)
    pm_scale = 66 / 34 * parallax_scale

    region["pmra_error_gaia_dr3"] = region["pmra_error_gaia_dr4"] * pm_scale
    region["pmdec_error_gaia_dr3"] = region["pmdec_error_gaia_dr4"] * pm_scale
    region["parallax_error_gaia_dr3"] = (
        region["parallax_error_gaia_dr4"] * parallax_scale
    )

    # Resample based on scale factors
    rng = np.random.default_rng(seed)

    region["pmra_gaia_dr3"] = rng.normal(
        loc=region["pmra_true"], scale=region["pmra_error_gaia_dr4"]
    )
    region["pmdec_gaia_dr3"] = rng.normal(
        loc=region["pmdec_true"], scale=region["pmdec_error_gaia_dr4"]
    )
    region["parallax_gaia_dr3"] = rng.normal(
        loc=region["parallax_true"], scale=region["parallax_error_gaia_dr4"]
    )

    return region


def _fetch_unique_locations(simulated_clusters):
    unique_locations = (
        simulated_clusters[
            ["l", "b", "distance", "pmra", "pmdec", "parallax", "path_region"]
        ]
        .drop_duplicates()
        .sort_values(["l", "b", "distance"])
        .reset_index(drop=True)
    )

    return unique_locations


def measure_cluster_densities(simulated_clusters):
    cluster_density_df = {}
    for i_row, a_row in simulated_clusters.iterrows():
        print(f"\r{i_row + 1} of {len(simulated_clusters)}", end="")
        members = pd.read_parquet(RESULTS_DIRECTORY / a_row["path"])
        cluster_density_df[i_row] = measure_cluster_density(a_row, members)

    print("")
    cluster_density_df = pd.DataFrame.from_dict(cluster_density_df, orient="index")
    simulated_clusters_with_density = simulated_clusters.join(cluster_density_df)
    return simulated_clusters_with_density


def measure_region_densities(simulated_clusters):
    unique_locations = _fetch_unique_locations(simulated_clusters)

    region_density_df = []
    paths = unique_locations["path_region"].unique()
    for i, short_path_region in enumerate(paths):
        print(f"\r{i + 1} of {len(paths)}", end="")

        valid_clusters = unique_locations.query(f"path_region=='{short_path_region}'")

        path_region = RESULTS_DIRECTORY / short_path_region
        path_metadata = path_region.parent / (path_region.stem + "_metadata.pickle")
        region = pd.read_parquet(path_region)
        region = add_gaia_dr3_uncertainties_based_on_sampling(region)

        with open(RESULTS_DIRECTORY / path_metadata, "rb") as file:
            region_area = pickle.load(file)["area"]

        region_density_df.append(
            measure_region_density(valid_clusters, region, region_area)
        )

    print("")
    region_density_df = pd.concat(region_density_df, ignore_index=True).drop(
        columns=["path_region", "parallax"]
    )

    return region_density_df


def measure_hr24_selection_function(simulated_clusters_full):
    n_stars_predictor = NStarsPredictor(models=100)
    density_estimator = GaiaDensityEstimator()

    coords = SkyCoord(
        ra=simulated_clusters_full["ra"].to_numpy() * u.deg,
        dec=simulated_clusters_full["dec"].to_numpy() * u.deg,
        pm_ra_cosdec=simulated_clusters_full["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=simulated_clusters_full["pmdec"].to_numpy() * u.mas / u.yr,
        distance=simulated_clusters_full["distance"].to_numpy() * u.pc,
        frame="icrs",
    )

    (
        simulated_clusters_full["n_stars_gaia_dr3_empirical"],
        simulated_clusters_full["med_error_gaia_dr3_empirical"],
    ) = n_stars_predictor(coords, simulated_clusters_full)

    simulated_clusters_full["rho_data_gaia_dr3_empirical"] = density_estimator(
        simulated_clusters_full["l"],
        simulated_clusters_full["b"],
        simulated_clusters_full["pmra"],
        simulated_clusters_full["pmdec"],
        1000 / simulated_clusters_full["distance"],
    )

    selection_function = HR24SelectionFunction()

    for m in ("gaia_dr3_empirical",):
        good_points = np.logical_and(
            np.isfinite(simulated_clusters_full[f"med_error_{m}"]),
            simulated_clusters_full[f"n_stars_{m}"] != 0,
        )
        simulated_clusters_full[f"probability_{m}"] = 0.0
        correction_factor = 1.0

        simulated_clusters_full.loc[good_points, f"probability_{m}"] = (
            selection_function(
                simulated_clusters_full.loc[good_points, f"rho_data_{m}"],
                simulated_clusters_full.loc[good_points, f"n_stars_{m}"],
                simulated_clusters_full.loc[good_points, f"med_error_{m}"] ** (1 / 3)
                / correction_factor,
                np.full(good_points.sum(), 3.0),
            )
        )

    return simulated_clusters_full


def add_final_detectability_ratios(simulated_clusters_full, missions_done):
    radius_angular = np.degrees(
        np.arctan(
            simulated_clusters_full["r_tidal"] / simulated_clusters_full["distance"]
        )
    )
    size_correction = np.pi * radius_angular**2

    for m in missions_done:
        if "empirical" in m:
            continue
        simulated_clusters_full[f"astrometric_ratio_{m}"] = (
            simulated_clusters_full[f"rho_astrometric_{m}"]
            + simulated_clusters_full[f"rho_data_pm_{m}"] * size_correction
        ) / (simulated_clusters_full[f"rho_data_pm_{m}"] * size_correction)
        simulated_clusters_full[f"position_ratio_{m}"] = (
            simulated_clusters_full[f"rho_position_{m}"]
            + simulated_clusters_full[f"rho_data_position_{m}"]
        ) / (simulated_clusters_full[f"rho_data_position_{m}"])
        simulated_clusters_full[f"total_ratio_{m}"] = (
            simulated_clusters_full[f"astrometric_ratio_{m}"]
            * simulated_clusters_full[f"position_ratio_{m}"]
        )


if __name__ == "__main__":
    simulated_clusters = pd.read_parquet(
        RESULTS_DIRECTORY / "simulated_clusters.parquet"
    )
    simulated_clusters["parallax"] = 1000 / simulated_clusters["distance"]

    print("Measuring cluster densities")
    simulated_clusters_with_density = measure_cluster_densities(simulated_clusters)

    print("Measuring region densities")
    region_density_df = measure_region_densities(simulated_clusters)
    simulated_clusters_full = simulated_clusters_with_density.merge(
        region_density_df, on=["l", "b", "distance", "pmra", "pmdec"], how="left"
    )

    print("Measuring HR24 selection function")
    missions_done = (
        "gaianir-l",
        "gaianir-m",
        "gaia_dr5",
        "gaia_dr4",
        "gaia_dr3",
        "gaia_dr3_empirical",
    )
    measure_hr24_selection_function(simulated_clusters_full)

    print("Performing final calculations before saving")

    add_final_detectability_ratios(simulated_clusters_full, missions_done)

    simulated_clusters_full.to_parquet(RESULTS_DIRECTORY / "detection_results.parquet")
