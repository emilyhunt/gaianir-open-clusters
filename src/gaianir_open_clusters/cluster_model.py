"""Main class defining an observation made with Gaia DR3."""

from __future__ import annotations
import ocelot.simulate.cluster
from ocelot.model.observation._base import (
    BaseObservation,
    BaseSelectionFunction,
)
from scipy.interpolate import interp1d

# from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
import numpy as np
import pandas as pd
import copy
from numpy.typing import ArrayLike
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy import units as u

from gaianir_open_clusters.gaia_nir_config import (
    GAIANIR_ANGULAR_RESOLUTION,
    COMBINE_GAIA_GAIANIR_ASTROMETRY,
    GAIA_GAIANIR_SEPARATION,
    FAINTEST_GAIA_MAGNITUDE_USED,
    FAINTEST_N_MAGNITUDE_USED,
)
from gaianir_open_clusters.photometry import (
    FILTERS,
    PhotometricModel,
    EXTINCTION_MODELS,
)
from gaianir_open_clusters.astrometry import (
    AstrometryModel,
    AstrometryModelElectronBased,
    combine_astrometry,
)
from gaianir_open_clusters.crowding import apply_cluster_crowding
from ocelot.model.observation.gaia.photutils import AG


_photometric_model_correct_band_names = PhotometricModel()
_photometric_model_correct_band_names.photometric_bands = [
    "gaianir_n_true",
    "gaianir_r_true",
    "gaianir_j_true",
    "gaianir_h_true",
    "gaianir_k_true",
]


class GaiaNIRObservationModel(BaseObservation):
    def __init__(
        self,
        mission_class="GaiaNIR-L",
        years=10,
        maximum_magnitude=FAINTEST_N_MAGNITUDE_USED,
        combined_astrometry=COMBINE_GAIA_GAIANIR_ASTROMETRY,
        combined_astrometry_separation=GAIA_GAIANIR_SEPARATION,
        combined_astrometry_max_gaia_mag=FAINTEST_GAIA_MAGNITUDE_USED,
    ):
        """A model for an observation made with Gaia DR3."""
        self.simulated_cluster = None  # To prevent it being removed # Todo: somehow stop issues with model not being re-assignable

        # Various stats about the model
        self.mission_class = mission_class
        self.years = years
        self.combined_astrometry = combined_astrometry
        self.maximum_magnitude = maximum_magnitude
        self.combined_astrometry_separation = (
            combined_astrometry_separation  # years between missions
        )
        self.combined_astrometry_max_gaia_mag = combined_astrometry_max_gaia_mag

    @property
    def name(self) -> str:
        """Type of observation modelled by this class.

        Should return a lowercase string, like 'gaia_dr3'.
        """
        combined = ""
        if self.combined_astrometry:
            combined = "-(combined)"

        return f"{self.mission_class.lower()}-{self.years}{combined}"

    @property
    def photometric_band_names(self) -> list[str]:
        """Names of photometric bands modelled by this system."""
        return ["gaianir_n", "gaianir_r", "gaianir_j", "gaianir_h", "gaianir_k"]

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

        # Todo add photometric errors
        return

    def apply_photometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        pass

    def calculate_astrometric_errors(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Calculate astrometric errors for a simulated cluster."""
        self._assert_simulated_cluster_not_reused(cluster)
        observation = cluster.observations[self.name]

        # Add some gaia-related things
        observation["g_effective_gaia"] = observation["gaia_dr3_g_true"] + AG(
            observation["extinction"], observation["temperature"]
        )
        observation["label"] = -1  # safe to ignore for Gaia astrometry

        # Calculate astrometric uncertainties
        if self.mission_class != "Gaia":
            gaia_nir_model = AstrometryModelElectronBased(
                self.mission_class, self.years
            )
            ra_error, dec_error, pmra_error, pmdec_error, parallax_error = (
                gaia_nir_model.predict(observation["gaianir_n"])
            )
        else:
            # Remove stars outside of the interpolation range
            # TODO this is repetitive and messy - can improve?
            good_stars = observation["g_effective_gaia"] < 25
            ra_error = np.full(len(good_stars), np.nan)
            dec_error = np.full(len(good_stars), np.nan)
            pmra_error = np.full(len(good_stars), np.nan)
            pmdec_error = np.full(len(good_stars), np.nan)
            parallax_error = np.full(len(good_stars), np.nan)

            gaia_model = AstrometryModel(self.mission_class, self.years)
            (
                ra_error[good_stars],
                dec_error[good_stars],
                pmra_error[good_stars],
                pmdec_error[good_stars],
                parallax_error[good_stars],
            ) = gaia_model.predict(
                observation.loc[good_stars],
                mag_column="g_effective_gaia",
                temperature_column="temperature",
            )

        # Optionally also combine astrometry from Gaia to the GaiaNIR observation
        if self.combined_astrometry and cluster.parameters.extinction < 25:
            gaia_model = AstrometryModel(mission="Gaia", years=10)
            good_stars = np.logical_and.reduce((
                observation["g_effective_gaia"] < self.combined_astrometry_max_gaia_mag,
                observation["g_effective_gaia"].notna(),
                observation["temperature"].notna(),
            ))
            if good_stars.sum() > 0:
                ra_error_past, dec_error_past = gaia_model.predict(
                    observation.loc[good_stars],
                    mag_column="g_effective_gaia",
                    temperature_column="temperature",
                )[:2]
                pmra_error[good_stars], pmdec_error[good_stars] = combine_astrometry(
                    ra_error[good_stars],
                    dec_error[good_stars],
                    ra_error_past,
                    dec_error_past,
                    separation=self.combined_astrometry_separation
                    + gaia_model.years / 2
                    + gaia_nir_model.years / 2,
                )

                observation = observation.drop(columns="label")

        observation["pmra_error"] = pmra_error
        observation["pmdec_error"] = pmdec_error
        observation["parallax_error"] = parallax_error

        cluster.observations[self.name] = observation

    def get_selection_functions(
        self, cluster: ocelot.simulate.cluster.SimulatedCluster
    ):
        """Get an initialized GaiaNIRSelectionFunction."""
        self._assert_simulated_cluster_not_reused(cluster)

        column = "gaianir_n"
        if self.mission_class == "Gaia":
            column = "g_effective_gaia"

        return [GaiaNIRSelectionFunction(self.maximum_magnitude, column=column)]

    def calculate_extinction(self, cluster: ocelot.simulate.cluster.SimulatedCluster):
        """Applies extinction in a given photometric band observed in this dataset."""
        # Remove stars too faint to see (things below PARSEC)
        # cluster.observations[self.name] = (
        #     cluster.observations[self.name]
        #     .loc[cluster.observations[self.name]["temperature"].notna()]
        #     .reset_index(drop=True)
        # )

        # Add photometry
        # Todo: ocelot: there should really be a make_photometry() function!
        true_bands = [f"{x}_true" for x in self.photometric_band_names]
        cluster.observations[self.name] = _photometric_model_correct_band_names.predict(
            cluster.observations[self.name],
            cluster.parameters.metallicity,
            luminosity_in_L_sun=True,
        )
        cluster.observations[self.name][true_bands] = (
            cluster.observations[self.name][true_bands]
            + 5 * np.log10(cluster.parameters.distance)
            - 5
        )

        # Then calculate reddening
        good_stars = cluster.observations[self.name]["temperature"].notna()
        query_data = pd.DataFrame.from_dict(
            {
                "A0": np.clip(
                    cluster.observations[self.name].loc[good_stars, "extinction"], 0, 50
                ),  # Clip as relation only valid to A_V=50
                "R0": 3.1,
                "teff": cluster.observations[self.name].loc[good_stars, "temperature"],
            }
        )
        for band in self.photometric_band_names:
            short_name = band.split("_")[1].upper()
            if short_name != "N":
                short_name = f"N_{short_name}"
            ax_over_a0 = EXTINCTION_MODELS[short_name].predict(query_data)
            cluster.observations[self.name].loc[good_stars, f"extinction_{band}"] = (
                cluster.observations[self.name].loc[good_stars, "extinction"]
                * np.asarray(ax_over_a0)
            )

        # cluster.observations[self.name] = cluster.observations[self.name]

    def calculate_resolving_power(
        self,
        primary: pd.DataFrame,
        secondary: pd.DataFrame,
        separation: Quantity,
    ) -> np.ndarray[float]:
        """Calculates the probability that a given pair of stars would be separately
        resolved."""
        separation = separation.to(u.arcsec).value
        return np.where(
            separation >= GAIANIR_ANGULAR_RESOLUTION[self.mission_class], 1.0, 0.0
        )

    def mag_to_flux(
        self, magnitude: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a magnitude in some band into a flux in some band."""
        magnitude = np.atleast_1d(magnitude)
        return 10 ** ((FILTERS[band].Vega_zero_mag - magnitude) / 2.5)

    def flux_to_mag(
        self, flux: int | float | ArrayLike, band: str
    ) -> int | float | ArrayLike:
        """Convert a flux in some band into a magnitude in some band."""
        flux = np.atleast_1d(flux)
        # We safely handle negative fluxes - they're set to inf
        good_fluxes = flux > 0
        magnitude = (
            -2.5 * np.log10(flux, where=good_fluxes) + FILTERS[band].Vega_zero_mag
        )
        magnitude[np.invert(good_fluxes)] = np.inf
        return magnitude  # todo actually always returns a np.ndarray

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
    def __init__(self, maximum_magnitude, column="gaianir_n"):
        """Gaia NIR selection function. Very approximate! For now, just assumes G>min is
        not observed.
        """
        self.maximum_magnitude = maximum_magnitude
        self.column = column

    def _query(self, observation: pd.DataFrame) -> np.ndarray:
        return (observation[self.column] < self.maximum_magnitude).astype(float)


_GOOD_COLUMNS = [
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "parallax",
    "parallax_error",
]

_EXTRA_GOOD_COLUMNS = [
    "ra",
    "dec",
    "pmra_true",
    "pmdec_true",
    "parallax_true",
    "l",
    "b",
    "mass",
    "temperature",
    "luminosity",
    "log_g",
    "gaianir_n",
    "gaianir_r",
    "gaianir_j",
    "gaianir_h",
    "gaianir_k",
    "g_effective_gaia",
]


def combine_observations(
    observations,
    crowding_metadata,
    main_star_columns=_EXTRA_GOOD_COLUMNS,
    observation_columns=_GOOD_COLUMNS,
):
    """Utility function that can combine observations from multiple ocelot cluster
    observations. This makes them a lot easier to save and load.

    Assumes that the gaianir-l observation is the first one.
    """
    # Prevents us from impacting the original object
    observations = copy.deepcopy(observations)

    keys = []

    merged_observation = None
    keys = []

    for name, observation in observations.items():
        name = (
            name.replace("-10-(combined)", "_combined")
            .replace("-10", "_dr5")
            .replace("-5", "_dr4")
        )
        short_name = name.replace("_combined", "")
        keys.append(short_name)

        if len(keys) == 0 and name != "gaianir-l_combined":
            raise ValueError("The first observation must always be gaianir-l_combined.")

        if name == "gaianir-l_combined":
            observation = observation[
                ["simulated_id"] + main_star_columns + observation_columns
            ]
        else:
            observation = observation[["simulated_id"] + observation_columns]

        observation[short_name] = True
        observation = observation.rename(
            columns={
                "pmra": f"pmra_{name}",
                "pmdec": f"pmdec_{name}",
                "parallax": f"parallax_{name}",
                "pmra_error": f"pmra_error_{name}",
                "pmdec_error": f"pmdec_error_{name}",
                "parallax_error": f"parallax_error_{name}",
            }
        )
        if merged_observation is None:
            merged_observation = observation
        else:
            merged_observation = merged_observation.merge(
                observation,
                on="simulated_id",
                how="left",  # Todo: this assumes gaianir-l is first! Be sure of this
            )
        merged_observation[short_name] = merged_observation[short_name] == True  # noqa: E712

    # Apply crowding
    key_to_mission = {
        "gaianir-l": "GaiaNIR-L",
        "gaianir-m": "GaiaNIR-M",
        "gaia_dr5": "Gaia",
        "gaia_dr4": "Gaia",
    }

    merged_observation = merged_observation.sort_values("gaianir_n").reset_index(
        drop=True
    )

    for key in keys:
        good_stars = merged_observation[key]
        merged_observation.loc[good_stars, key] = apply_cluster_crowding(
            merged_observation.loc[good_stars],
            crowding_metadata,
            key_to_mission[key],
            drop_stars=False,
        )

    return merged_observation.sort_values("simulated_id").reset_index(drop=True)

    # merged_observation = (
    #     merged_observation.loc[np.any(merged_observation[keys], axis=1)]
    #     .sort_values("simulated_id")
    #     .reset_index(drop=True)
    # )
