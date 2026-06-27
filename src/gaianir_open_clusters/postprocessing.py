"""Helpers for calculating detectability and other post-processing steps."""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from gaianir_open_clusters.util import position_to_max_galaxy_distance


_default_missions = (
    "gaianir-l",
    "gaianir-m",
    "gaia_dr5",
    "gaia_dr4",
    "gaia_dr3",
    "gaia_dr3_empirical",
)


def assign_detection_probabilities(
    detection_results, minimum_stars=100, minimum_ratio=5, missions=_default_missions
):
    """Converts 5D density and number of stars into a detection probability (1 or 0)
    based on assigned settings.
    """
    for m in _default_missions:
        if "empirical" in m:
            continue

        good_ratio = detection_results[f"total_ratio_{m}"] > minimum_ratio
        detection_results[f"probability_{m}"] = np.where(
            detection_results[f"n_stars_{m}"] > minimum_stars,
            (good_ratio).astype(float),
            0.0,
        )

    return detection_results


def calculate_detection_distances(
    detection_results,
    cluster,
    empirical_probability=0.5,
    return_as_xy=True,
    missions=_default_missions,
    max_galaxy_distance=20000,
):
    """Calculates the distances to which something can be detected."""

    l_values = np.unique(detection_results["l"])
    detectabilities = {mission: [] for mission in missions}
    for l in l_values:
        subsample = detection_results.query(f"cluster=='{cluster}' and l=={l}")
        max_distance = position_to_max_galaxy_distance(l, max_galaxy_distance)

        for m in missions:
            distances, probabilities = (
                subsample[["distance", f"probability_{m}"]].to_numpy().T
            )
            is_below = probabilities < empirical_probability
            false_values = is_below.nonzero()[0]

            # If everything is detected, just set to max distance
            if len(false_values) == 0:
                detectabilities[m].append(max_distance)
                continue

            # If it's never detected, set to zero
            first_false_value = false_values[0]
            if first_false_value == 0:
                detectabilities[m].append(0)
                continue

            # Otherwise, interpolate between the two sides
            slicer = slice(first_false_value - 1, first_false_value + 1)
            detectabilities[m].append(
                np.clip(
                    np.interp(
                        empirical_probability,
                        probabilities[slicer][::-1],
                        distances[slicer][::-1],
                    ),
                    0,
                    max_distance,
                )
            )

    if return_as_xy:
        return detectabilities_to_xy(detectabilities, l_values)
    return detectabilities, l_values


def detectabilities_to_xy(detectabilities, l_values):
    xy_values = {}
    for m in detectabilities.keys():
        coord = (
            SkyCoord(
                l=np.asarray(l_values) * u.deg,
                b=np.zeros_like(l_values) * u.deg,
                distance=np.asarray(detectabilities[m]) * u.pc,
                frame="galactic",
            )
            .transform_to("galactocentric")
            .represent_as("cartesian")
        )  # type: ignore  (fuck pylance)
        x_values = coord.x.value
        y_values = coord.y.value
        x_values = np.append(x_values, x_values[0])
        y_values = np.append(y_values, y_values[0])
        xy_values[m] = [x_values, y_values]
    return xy_values
