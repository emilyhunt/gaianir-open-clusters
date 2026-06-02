"""Simple interface to GaiaNIR astrometric uncertainty models."""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from gaianir_open_clusters.gaia_nir_config import ASTROMETRIC_DATA


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
        data_ms = pd.concat([data[type] for type in _main_sequence], ignore_index=True)
        data_giant = pd.concat(
            [data[type] for type in _main_sequence], ignore_index=True
        )

        self._main_sequence_interpolator = LinearNDInterpolator(
            data_ms[["G", "teff"]].to_numpy(), data_ms["parallax_error"], rescale=True
        )
        self._giant_interpolator = LinearNDInterpolator(
            data_giant[["G", "teff"]].to_numpy(),
            data_giant["parallax_error"],
            rescale=True,
        )

    def predict(self, data):
        is_giant = data["label"] >= 2
        is_ms = np.invert(is_giant)

        to_use = data[["G", "teff"]].to_numpy()
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

        # Also add proper motion errors, given scaling relations in Hobbs+ (in prep.)
        # position_error = 0.75 * parallax_error

        # Todo I am not quite sure how to get to pmra error from parallax error - check this!
        pmra_error = parallax_error / (self.years / 2.5)
        pmdec_error = pmra_error

        return pmra_error, pmdec_error, parallax_error
