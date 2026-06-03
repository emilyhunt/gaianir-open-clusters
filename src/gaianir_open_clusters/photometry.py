"""GaiaNIR photometric information, courtesy of the wonderful pyphot library."""

import pyphot
import numpy as np
from pyphot.config import units
from gaianir_open_clusters.gaia_nir_config import GAIA_NIR_FILTERS
from gaianir_open_clusters.config import RESULTS_DIRECTORY
from scipy.interpolate import interp1d, LinearNDInterpolator
import warnings
from pathlib import Path
import pandas as pd
import astropy.units as u
from astropy import constants
from dustapprox.models import PolynomialModel


# Initialize all GaiaNIR filters & corresponding information
FILTERS = {}

for filter, values in GAIA_NIR_FILTERS.items():
    half_width = values["width"] / 2
    wavelengths = np.asarray(
        [
            values["mid"] - half_width - 0.0000001,
            values["mid"] - half_width,
            values["mid"],
            values["mid"] + half_width,
            values["mid"] + half_width + 0.0000001,
        ]
    )
    GAIA_NIR_FILTERS[filter]["transmission"] = np.asarray([0.0] + [0.7] * 3 + [0.0])

    FILTERS[filter] = pyphot.Filter(
        wavelengths * units.U("nm"),
        np.full_like(wavelengths, GAIA_NIR_FILTERS[filter]["transmission"]),
        name=filter,
        dtype="photon",
        vega="stis_011",  # Specify the Vega flavor,
    )


class PhotometricModel:
    def __init__(self):
        _path_to_photometric_data = RESULTS_DIRECTORY / "photometry.parquet"
        self.fitted = True
        if not _path_to_photometric_data.exists():
            warnings.warn(
                f"Photometric data file ({_path_to_photometric_data}) does not exist. "
                "Photometric interpolation is only possible after running the first few"
                " scripts, which generate this data."
            )
            self.fitted = False
            return

        # Grab data
        self._photometric_grid = pd.read_parquet(_path_to_photometric_data)

        # Specify some regions to clip teff and logg against
        self._min_teff = self._photometric_grid["teff"].min()
        self._max_teff = self._photometric_grid["teff"].max()

        log_g_stats = (
            self._photometric_grid.groupby("teff")
            .aggregate(min=pd.NamedAgg("logg", "min"), max=pd.NamedAgg("logg", "max"))
            .reset_index()
        )
        self._min_log_g_int = interp1d(
            log_g_stats["teff"], log_g_stats["min"], bounds_error=True
        )
        self._max_log_g_int = interp1d(
            log_g_stats["teff"], log_g_stats["max"], bounds_error=True
        )

        # Then setup the interpolator
        self.columns = ["teff", "logg", "feh"]
        self.photometric_bands = list(GAIA_NIR_FILTERS.keys())
        x = self._photometric_grid[self.columns].to_numpy()
        y_big = self._photometric_grid[self.photometric_bands]
        self._interpolator = LinearNDInterpolator(x, y_big)

    def predict(self, data, metallicity, luminosity_in_L_sun=False):
        """Predict GaiaNIR (absolute) photometry in all bands."""
        # Some initial calculations
        # data["teff"] = 10 ** data["logTe"]
        # data["lum"] = 10 ** data["logL"] * constants.L_sun.value
        luminosity_multiplier = 1
        if luminosity_in_L_sun:
            luminosity_multiplier = constants.L_sun.value

        data["radius"] = np.sqrt(
            data["luminosity"] * luminosity_multiplier
            / (4 * np.pi * constants.sigma_sb.value * data["temperature"] ** 4)
        )

        # Grab radius-free photometry
        x = data[["temperature", "log_g"]].copy()
        x["metallicity"] = metallicity

        x["temperature"] = np.clip(x["temperature"], self._min_teff, self._max_teff)
        x["log_g"] = np.clip(
            x["log_g"],
            self._min_log_g_int(x["temperature"]),
            self._max_log_g_int(x["temperature"]),
        )
        result = self._interpolator(x.to_numpy())

        # Apply stellar radius correction
        # (magnitudes from pyphot are at the stellar surface)
        radii_offset = 5 * np.log10(
            (data["radius"].to_numpy() / (10 * u.pc).to(u.m)).value
        ).reshape(-1, 1)

        # Apply a bad temperature correction to any stars with clipped teff
        temperature_offset = 4 * np.log10(
            x["temperature"] / data["temperature"]
        ).to_numpy().reshape(-1, 1)

        # Assign & return
        result = result - radii_offset + temperature_offset

        # for i, band in enumerate(self.photometric_bands):
        data[self.photometric_bands] = result
        return data


PHOTOMETRY_PREDICTOR = PhotometricModel()

EXTINCTION_MODELS = {}
for band in "N", "N_R", "N_J", "N_H", "N_K":
    EXTINCTION_MODELS[band] = PolynomialModel.from_file(
        RESULTS_DIRECTORY / "extinction_model.ecsv", band
    )
