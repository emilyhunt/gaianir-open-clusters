"""Main class defining an observation made with Gaia DR3."""

from __future__ import annotations
from ocelot.model.observation._base import (
    BaseObservation,
    BaseSelectionFunction,
)
import ocelot.simulate.cluster
from scipy.interpolate import interp1d

# from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy import units as u

from gaianir_open_clusters.gaia_nir_config import GAIA_NIR_MINIMUM_SEPARATION


class GaiaNIRObservationModel(BaseObservation):
    def __init__(self, minimum_magnitude=24):
        """A model for an observation made with Gaia DR3."""
        self.simulated_cluster = None  # To prevent it being removed # Todo: somehow stop issues with model not being re-assignable
        self.minimum_magnitude = minimum_magnitude

    @property
    def name(self) -> str:
        """Type of observation modelled by this class.

        Should return a lowercase string, like 'gaia_dr3'.
        """
        return "gaia_dr3"

    @property
    def photometric_band_names(self) -> list[str]:
        """Names of photometric bands modelled by this system."""
        return ["gaia_nir_n", "gaia_nir_r", "gaia_nir_j", "gaia_nir_h", "gaia_nir_k"]

    @property
    def has_proper_motions(self) -> bool:
        return True

    @property
    def has_parallaxes(self) -> bool:
        return True

    def calculate_photometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Calculate photometric errors for a simulated cluster."""
        self._assert_simulated_cluster_not_reused(cluster)
        if self.matching_stars is None:
            self.matching_stars, self.stars_to_assign = _closest_gaia_star(
                cluster.observations["gaia_dr3"], self.representative_stars
            )

        for band in ("g", "bp", "rp"):
            cluster.observations["gaia_dr3"].loc[
                self.stars_to_assign, f"gaia_dr3_{band}_flux_error"
            ] = self.matching_stars[f"phot_{band}_mean_flux_error"].to_numpy()

    def apply_photometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Custom method to apply photometric errors to a simulated cluster.

        Method incorporates the underestimated BP and RP flux measurement issue in DR3.

        Follows things discussed in Riello+21, section 8.1.
        """
        observation = cluster.observations["gaia_dr3"]

        # Calculate true flux
        fluxes = {}
        for band in self.photometric_band_names:
            fluxes[band] = self.mag_to_flux(observation[band].to_numpy(), band)

        # For BP and RP, count how many times Gaia would have observed the star, and
        # then reverse-apply the flux calculation mistake in Gaia DR3
        if self.overestimate_bp_rp_fluxes:
            for band in ["gaia_dr3_bp", "gaia_dr3_rp"]:
                fluxes = self._apply_incorrect_flux_summing_to_flux(
                    observation, fluxes, band
                )

        # Now, finally, we can apply photometric errors from other sources & move on!
        for band in self.photometric_band_names:
            new_fluxes = cluster.random_generator.normal(
                loc=fluxes[band],
                scale=observation[f"{band}_flux_error"].to_numpy(),
            )
            observation[band] = self.flux_to_mag(new_fluxes, band)

    def calculate_astrometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Calculate astrometric errors for a simulated cluster."""
        self._assert_simulated_cluster_not_reused(cluster)
        if self.matching_stars is None:
            self.matching_stars, self.stars_to_assign = _closest_gaia_star(
                cluster.observations["gaia_dr3"], self.representative_stars
            )

        for column in ("pmra_error", "pmdec_error", "parallax_error"):
            cluster.observations["gaia_dr3"].loc[self.stars_to_assign, column] = (
                self.matching_stars[column].to_numpy()
            )

    def get_selection_functions(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Get an initialized GaiaNIRSelectionFunction."""
        self._assert_simulated_cluster_not_reused(cluster)
        return [GaiaNIRSelectionFunction()]

    def calculate_extinction(self, cluster: ocelot.simulate.cluster.SimulatedCluster):
        """Applies extinction in a given photometric band observed in this dataset."""
        observation = cluster.observations["gaia_dr3"]

        for band, func in zip(self.photometric_band_names, (AG, ABP, ARP)):
            observation[f"extinction_{band}"] = func(
                observation["extinction"], observation["temperature"]
            )

    def calculate_resolving_power(
        self,
        primary: pd.DataFrame,
        secondary: pd.DataFrame,
        separation: Quantity,
    ) -> np.ndarray[float]:
        """Calculates the probability that a given pair of stars would be separately
        resolved."""
        separation = separation.to(u.arcsec).value
        return np.where(separation >= GAIA_NIR_MINIMUM_SEPARATION, 1.0, 0.0)

    def mag_to_flux(
        self, magnitude: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a magnitude in some band into a flux in some band."""
        self._check_band_name(band)
        magnitude = np.atleast_1d(magnitude)
        return 10 ** (
            (self.ZEROPOINTS[band] - magnitude) / 2.5
        )  # todo actually always returns a np.ndarray

    def flux_to_mag(
        self, flux: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a flux in some band into a magnitude in some band."""
        self._check_band_name(band)
        flux = np.atleast_1d(flux)
        # We safely handle negative fluxes - they're set to inf
        good_fluxes = flux > 0
        magnitude = -2.5 * np.log10(flux, where=good_fluxes) + self.ZEROPOINTS[band]
        magnitude[np.invert(good_fluxes)] = np.inf
        return magnitude  # todo actually always returns a np.ndarray

    def _check_band_name(self, band: str):
        if band not in self.ZEROPOINTS:
            raise ValueError(
                f"band {band} is not the correct name of a photometric band modelled "
                "in this observation."
            )

    def _assert_simulated_cluster_not_reused(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        # Todo this is bad and should be improved lol - model should be cluster-agnostic
        if self.simulated_cluster is None:
            self.simulated_cluster = cluster
            return
        if cluster is not self.simulated_cluster:
            raise RuntimeError(
                "Gaia DR3 model may not be reused on different clusters!"
            )


class GaiaNIRSelectionFunction(BaseSelectionFunction):
    def __init__(self, minimum_magnitude):
        """Gaia NIR selection function. Very approximate! For now, just assumes G>min is
        not observed.
        """
        self.minimum_magnitude = minimum_magnitude

    def _query(self, observation: pd.DataFrame) -> np.ndarray:
        return (observation["gaia_nir_n"] < self.minimum_magnitude).astype(float)
