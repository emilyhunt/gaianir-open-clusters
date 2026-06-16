"""Utilities to apply crowding to GaiaNIR observations."""

import numpy as np
import pandas as pd
from gaianir_open_clusters.gaia_nir_config import GAIANIR_ANGULAR_RESOLUTION
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

    important_things = dict(area=area, magnitude_ppfunc=magnitude_ppfunc)
    for mission, resolution in GAIANIR_ANGULAR_RESOLUTION.items():
        density_param = _get_poisson_param(region, area, resolution)
        region[f"uncrowded_{mission.lower()}"] = _sample_background_crowding(
            region["N"].to_numpy(), density_param, magnitude_ppfunc, rng
        )
        important_things[f"density_param_{mission.lower()}"] = density_param

    return region, important_things


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

    good_cluster_stars = _sample_background_crowding(
        cluster["gaianir_n"].to_numpy(),
        poisson_param,
        crowding_metadata["magnitude_ppfunc"],
        rng,
    )
    good_cluster_stars_self = _radius_crowding(cluster, resolution)

    good_stars = np.logical_and(good_cluster_stars, good_cluster_stars_self)
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
