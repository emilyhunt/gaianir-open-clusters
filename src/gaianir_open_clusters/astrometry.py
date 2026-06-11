"""Simple interface to GaiaNIR astrometric uncertainty models."""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interp1d
from gaianir_open_clusters.gaia_nir_config import ASTROMETRIC_DATA
from gaianir_open_clusters.photometry import FILTERS


# N.B. may break if the astrometric data model changes format significantly
_stellar_types = [folder.stem[2:] for folder in ASTROMETRIC_DATA.glob("*")]
_main_sequence = [star for star in _stellar_types if "III" in star]
_giant = [star for star in _stellar_types if "V" in star]

_main_sequence = _main_sequence + ["B1V"]  # Add B giant, as it's so close to B-type MS
_giant = _giant + ["M5III"]  # So that very red giant stars have something

# from:
# https://www.not.iac.es/instruments/notcam/ReferenceInfo/temp.html
# https://about.ifa.hawaii.edu/ukirt/calibration-and-standards/astronomical-utilities/temperatures-and-colors-of-stars/
_effective_temperature_map = {
    "B1V": 25600,
    "G2V": 5830,
    "M0V": 3800,
    "A5III": 8160,  # about right..
    "K5III": 3980,
    "M5III": 3420,
    "M10III": 2300,  # guess
}


class AstrometryModel:
    def __init__(self, mission="GaiaNIR-L", years=10):
        if not isinstance(mission, int):
            if mission == "Gaia":
                mission = 1
            elif mission == "GaiaNIR-M":
                mission = 2
            elif mission == "GaiaNIR-L":
                mission = 3
            else:
                raise ValueError("'mission' kwarg not recognized")

        self.mission = mission
        self.years = years
        data = self._read_astrometric_data()

        # Setup some ranges
        main_sequence_temps = [
            _effective_temperature_map[star] for star in _main_sequence
        ]
        giant_temps = [_effective_temperature_map[star] for star in _main_sequence]
        self.main_sequence_temp_range = (
            np.min(main_sequence_temps),
            np.max(main_sequence_temps),
        )
        self.giant_temp_range = (np.min(giant_temps), np.max(giant_temps))

        data_example = data[list(data.keys())[0]]
        self.g_range = (data_example["G"].min(), data_example["G"].max())

        self._build_astrometric_uncertainty_interpolators(data)

    def _read_astrometric_data(self):
        required_files = [
            ASTROMETRIC_DATA / f"uk{star}/M{self.mission}Yr{self.years}G.dat"
            for star in _stellar_types
        ]

        data = {}

        for file, star in zip(required_files, _stellar_types):
            data[star] = pd.read_csv(
                file, delimiter=r"\s+", names=["G", "parallax_error", "electrons"]
            )
            data[star]["teff"] = _effective_temperature_map[star]

        return data

    def _build_astrometric_uncertainty_interpolators(self, data):
        data_ms = (
            pd.concat([data[type] for type in _main_sequence], ignore_index=True)
            .sort_values(["G", "teff"])
            .reset_index(drop=True)
        )
        data_giant = (
            pd.concat([data[type] for type in _main_sequence], ignore_index=True)
            .sort_values(["G", "teff"])
            .reset_index(drop=True)
        )

        ms_grid = data_ms["G"].unique(), data_ms["teff"].unique()
        giant_grid = (data_giant["G"].unique(), data_giant["teff"].unique())

        ms_y = (
            data_ms["parallax_error"]
            .to_numpy()
            .reshape(ms_grid[0].size, ms_grid[1].size)
        )
        giant_y = (
            data_giant["parallax_error"]
            .to_numpy()
            .reshape(giant_grid[0].size, giant_grid[1].size)
        )

        self._main_sequence_interpolator = RegularGridInterpolator(ms_grid, ms_y)
        self._giant_interpolator = RegularGridInterpolator(giant_grid, giant_y)

    def predict(self, data, mag_column="G", temperature_column="teff"):
        is_giant = data["label"] >= 2
        is_ms = np.invert(is_giant)

        to_use = data[[mag_column, temperature_column]].to_numpy()
        to_use[is_giant, 1] = np.clip(
            to_use[is_giant, 1], self.giant_temp_range[0], self.giant_temp_range[1]
        )
        to_use[is_ms, 1] = np.clip(
            to_use[is_ms, 1],
            self.main_sequence_temp_range[0],
            self.main_sequence_temp_range[1],
        )

        parallax_error = np.zeros(len(data))
        parallax_error[is_giant] = self._giant_interpolator(to_use[is_giant])
        parallax_error[is_ms] = self._main_sequence_interpolator(to_use[is_ms])

        # Convert to mas
        parallax_error = parallax_error / 1000

        # Also add proper motion errors, given scaling relations in Hobbs+ (in prep.)
        # and at https://www.cosmos.esa.int/web/gaia/science-performance
        ra_error = parallax_error * 0.8
        dec_error = parallax_error * 0.7

        pmra_error = parallax_error * 0.29 * (10 / self.years)
        pmdec_error = pmra_error * 0.25 * (10 / self.years)

        return ra_error, dec_error, pmra_error, pmdec_error, parallax_error


class AstrometryModelElectronBased:
    def __init__(self, mission="GaiaNIR-L", years=10):
        if not isinstance(mission, int):
            if mission == "Gaia":
                raise ValueError(
                    "The electron-based model is not correctly calibrated for Gaia - "
                    "only GaiaNIR. Please just use the simpler AstrometryModel instead."
                )
            elif mission == "GaiaNIR-M":
                mission = 2
            elif mission == "GaiaNIR-L":
                mission = 3
            else:
                raise ValueError("'mission' kwarg not recognized")
        self.mission = mission
        self.years = years
        electrons, parallax_error = self._read_astrometric_data()

        self._interpolator = interp1d(
            electrons,
            parallax_error,
            bounds_error=False,
            fill_value=(parallax_error.max(), parallax_error.min()),
        )

    def _read_astrometric_data(self):
        file = ASTROMETRIC_DATA / f"ukB1V/M{self.mission}Yr{self.years}G.dat"
        data = pd.read_csv(
            file, delimiter=r"\s+", names=["G", "parallax_error", "electrons"]
        )
        return data["electrons"].to_numpy(), data["parallax_error"].to_numpy()

    def predict(self, magnitude):
        electrons = self.mag_to_electrons(magnitude)

        parallax_error = self._interpolator(electrons) / 1000

        # Also add proper motion errors, given scaling relations in Hobbs+ (in prep.)
        # and at https://www.cosmos.esa.int/web/gaia/science-performance
        ra_error = parallax_error * 0.8
        dec_error = parallax_error * 0.7

        pmra_error = parallax_error * 0.29 * (10 / self.years)
        pmdec_error = pmra_error * 0.25 * (10 / self.years)

        return ra_error, dec_error, pmra_error, pmdec_error, parallax_error

    def mag_to_electrons(self, mag):
        # Magic number that calibrates from David's GaiaNIR sims (which include e.g. the
        # telescope size, scan rate, averaging, and all sorts) against my simple
        # photometric estimations
        # Todo: this ought to be a data-driven number
        calibration_factor = 427.54392151861185 / 10.738344786983658

        if self.mission == 2:
            calibration_factor /= 2

        return 10 ** ((mag - FILTERS["N"].Vega_zero_mag) / -2.5) * calibration_factor


def combine_astrometry(
    ra_error_1,
    dec_error_1,
    ra_error_2,
    dec_error_2,
    separation,
):
    pmra_error_combined = (ra_error_1**-2 + ra_error_2**-2) ** -0.5 / separation
    pmdec_error_combined = (dec_error_1**-2 + dec_error_2**-2) ** -0.5 / separation

    return pmra_error_combined, pmdec_error_combined
