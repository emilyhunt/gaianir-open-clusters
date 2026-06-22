"""Utilities to apply crowding to GaiaNIR observations."""

import numpy as np
import pandas as pd
from gaianir_open_clusters.gaia_nir_config import (
    GAIANIR_ANGULAR_RESOLUTION,
    GAIANIR_MAXIMUM_STARS_PER_SQUARE_DEGREE,
)
from scipy.stats import poisson, ecdf
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors


def apply_background_crowding(
    region: pd.DataFrame,
    area: float,
    seed=None,
):
    """Applies crowding to a background region and generates required distributions for
    working with it.
    """
    rng = np.random.default_rng(seed)
    magnitude_ppfunc = _setup_background_crowding_stats(region)

    crowding_metadata = dict(area=area, magnitude_ppfunc=magnitude_ppfunc)
    for mission, resolution in GAIANIR_ANGULAR_RESOLUTION.items():
        detected = f"uncrowded_{mission.lower()}"
        density_param = _get_poisson_param(region, area, resolution)
        region[detected] = _sample_background_crowding(
            region["N"].to_numpy(), density_param, magnitude_ppfunc, rng
        )
        crowding_metadata[f"density_param_{mission.lower()}"] = density_param

        good_mags, max_magnitude_to_transmit = _calculate_transmission_crowding(
            region, area, mission, detected
        )
        region.loc[region[detected], detected] = good_mags
        crowding_metadata[f"max_magnitude_to_transmit_{mission}"] = (
            max_magnitude_to_transmit
        )

    return region, crowding_metadata


def apply_cluster_crowding(
    cluster: pd.DataFrame,
    crowding_metadata: dict,
    mission: str,
    seed=None,
    drop_stars: bool = True,
):
    """Assumes that cluster is sorted by magnitude!"""
    rng = np.random.default_rng(seed)
    resolution = GAIANIR_ANGULAR_RESOLUTION[mission]
    poisson_param = crowding_metadata[f"density_param_{mission.lower()}"]

    # Apply crowding due to background
    good_cluster_stars = _sample_background_crowding(
        cluster["gaianir_n"].to_numpy(),
        poisson_param,
        crowding_metadata["magnitude_ppfunc"],
        rng,
    )
    good_cluster_stars_self = _radius_crowding(cluster, resolution)

    good_stars = np.logical_and(good_cluster_stars, good_cluster_stars_self)

    # Apply additional crowding due to transmission limitations
    max_mag_transmit = crowding_metadata[f"max_magnitude_to_transmit_{mission}"]
    if max_mag_transmit < cluster['gaianir_n'].max():
        good_stars[good_stars] = cluster.loc[good_stars, "gaianir_n"] < max_mag_transmit

    if not drop_stars:
        return good_stars

    return cluster.loc[good_stars].reset_index(drop=True)


def _setup_background_crowding_stats(region: pd.DataFrame):
    # For sampling random magnitudes:
    # Make empirical CDF of magnitudes in the region
    empirical_cdf = ecdf(region["N"])
    quantiles, cdf = empirical_cdf.cdf.quantiles, empirical_cdf.cdf.probabilities

    # Zero-pad the start
    quantiles = np.append(quantiles[0] - (quantiles[1] - quantiles[0]), quantiles)
    cdf = np.append(0.0, cdf)

    # Create percentile point function so it can be sampled
    ppfunc = interp1d(cdf, quantiles, bounds_error=True)

    return ppfunc


def _get_poisson_param(region, area_degrees_sq, resolution_radians):
    # Setup a distribution to guess how many points there are
    density = len(region) / area_degrees_sq
    volume_obs = np.pi * np.degrees(resolution_radians) ** 2
    return density * volume_obs


def _sample_background_crowding(
    magnitudes, poisson_param, background_magnitude_ppfunc, rng
):
    # Sample number of neighbors in background
    background_dist = poisson(poisson_param)
    n_neighbors = background_dist.rvs(size=len(magnitudes), random_state=rng)
    could_be_removed = n_neighbors != 0

    # For those with neighbors, assign them a max neighbor magnitude
    brightest_neighbor_magnitude = np.asarray(
        [
            np.min(background_magnitude_ppfunc(rng.uniform(size=n)))
            for n in n_neighbors[could_be_removed]
        ]
    )

    # Stars are removed if they don't beat this
    good_stars = np.ones(len(magnitudes), dtype=bool)
    good_stars[could_be_removed] = (
        magnitudes[could_be_removed] < brightest_neighbor_magnitude
    )
    return good_stars


def _calculate_transmission_crowding(region, area, mission, detected):
    """Applies crowding due to the transmission limitations of Gaia/GaiaNIR, which limit
    it to a maximum number of stars per square degree. For Gaia, this is ~1.4 million
    (see Cantat-Gaudin+23); for GaiaNIR, this is assumed to be a lot higher.
    """
    n_detected_stars = region[detected].sum()
    stars_per_square_degree = n_detected_stars / area
    max_stars = GAIANIR_MAXIMUM_STARS_PER_SQUARE_DEGREE[mission]
    fraction_of_stars_to_keep = np.clip(max_stars / stars_per_square_degree, 0.0, 1.0)

    good_mags = np.ones(n_detected_stars, dtype=bool)
    max_magnitude_to_transmit = 99
    if fraction_of_stars_to_keep < 1.0:
        magnitudes = region.loc[region[detected], "N"].to_numpy()
        sorted_mags = np.sort(magnitudes)
        index_max = int(np.round(n_detected_stars * fraction_of_stars_to_keep))
        max_magnitude_to_transmit = sorted_mags[index_max]
        good_mags = magnitudes < max_magnitude_to_transmit

    return good_mags, max_magnitude_to_transmit


def _radius_crowding(cluster: pd.DataFrame, resolution_radians: float):
    # Find nearest neighbors of every point
    values = np.radians(cluster[["l", "b"]].to_numpy())
    neighbor_estimator = NearestNeighbors(
        radius=resolution_radians, metric="haversine", n_jobs=1
    ).fit(values)
    distances, indices = neighbor_estimator.radius_neighbors(values, sort_results=True)

    # Iterate over every star in brightness order, removing all stars that it neighbors
    # that are fainter than it.
    good_stars = np.ones(len(cluster), dtype=bool)
    bad_stars = set()

    for i_star in range(len(good_stars)):
        # No neighbors
        if len(indices[i_star]) == 1:
            continue

        # Already yeeted
        if i_star in bad_stars:
            continue

        # Has neighbors - since we have a sorted dataframe and a distance-sorted indices
        # list, we actually only need to do other stars. The current star is always the
        # brightest of the few =)
        bad_stars.update(indices[i_star][1:])

    good_stars[list(bad_stars)] = False
    return good_stars
