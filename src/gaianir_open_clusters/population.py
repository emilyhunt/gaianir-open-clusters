"""A little wrapper for Synthpop to make simulating a region easier.

Should need basically no changing once defaults are set up.
"""

from __future__ import annotations
import synthpop
import numpy as np
import pandas as pd
from gaianir_open_clusters.config import (
    SYNTHPOP_DEFAULT_CONFIG,
    SYNTHPOP_GAIANIR_CONFIG,
    DATA_DIRECTORY,
)
from gaianir_open_clusters.photometry import PHOTOMETRY_PREDICTOR, EXTINCTION_MODELS
from gaianir_open_clusters.astrometry import (
    AstrometryModelElectronBased,
    AstrometryModel,
    combine_astrometry,
)
from gaianir_open_clusters.gaia_nir_config import (
    FAINTEST_N_MAGNITUDE,
    FAINTEST_GAIA_MAGNITUDE,
    FAINTEST_GAIA_MAGNITUDE_USED,
    GAIA_GAIANIR_SEPARATION,
)
from gaianir_open_clusters.crowding import apply_background_crowding
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential
from astropy import units as u
from dustmaps.bayestar import BayestarQuery
from dustmaps.decaps import DECaPSQueryLite
from synthpop.synthpop_utils.synthpop_logging import logger
import logging

import os

os.environ["OCELOT_DATA"] = (DATA_DIRECTORY / "ocelot_data").as_posix()

import ocelot.simulate.cluster  # noqa: F401
from ocelot.model.observation.gaia.photutils import AG


def switch_off_synthpop_logging():
    """It is SO HARD to make synthpop quiet"""

    def dummy_func(*args, **kwargs):
        pass

    logger.setLevel(logging.ERROR)
    logger.stream_logger.setLevel(logging.ERROR)
    # logger.filelogger.setLevel(logging.ERROR)
    logger.debugger.setLevel(logging.ERROR)
    logger.create_info_section = dummy_func
    logger.create_info_subsection = dummy_func


switch_off_synthpop_logging()
_model = synthpop.SynthPop(
    specific_config=SYNTHPOP_GAIANIR_CONFIG,
    default_config=SYNTHPOP_DEFAULT_CONFIG,
    # extinction_map_kwargs=dict(name="no_extinction"),
)
_bayestar_map = BayestarQuery(max_samples=1)
_zucker_map = DECaPSQueryLite(mean_only=True)


def simulate_region(l, b, area, minimum_stars=1000):
    """Simulate a small patch of the Milky Way to use for data density calculations."""

    if not _model.populations_are_initialized:
        print("Initializing populations for first time...")
        _model.init_populations()

    # Blank objects to fucking shut the fuck up pylance UGH
    result = pd.DataFrame()
    coords = SkyCoord(ra=[], dec=[], unit="deg")

    attempt = 1
    while len(result) < minimum_stars:
        if attempt > 1:
            area = _inflate_area(area, len(result), minimum_stars)

        # PART 1: try simulating the region, be sure that it has enough stars
        print(
            f"Simulating region... (attempt {attempt}, area {area * 60**2:.3f} arcmin^2)"
        )
        result = _model.process_location(
            l_deg=l, b_deg=b, solid_angle=area, save_data=False
        )[0]

        if len(result) < minimum_stars:
            print(f"  trying again; only have {len(result)} stars")
            attempt += 1
            continue

        # PART 2: simulate the region's photometry
        print(f"  now adding photometry for {len(result)} stars...")

        # Somehow not everything has a temperature assigned??? lmao isochrones who are they
        result = result.loc[result["logTeff"].notna()].reset_index(drop=True)

        coords = _get_skycoord(result)
        result = _calculate_photometry(result, coords)
        _assign_true_astrometry(result, coords)

        result = result.loc[result["N"] < FAINTEST_N_MAGNITUDE].reset_index(drop=True)

        if len(result) < minimum_stars:
            print(f"  trying again; only have {len(result)} stars")
            attempt += 1

    print(f"Success! Calculating astrometry for {len(result)} remaining stars.")
    _calculate_gaianir_astrometry(result)
    _calculate_gaia_astrometry(result)
    _calculate_combined_astrometry(result)
    _sample_astrometry(result)

    print("Applying crowding.")
    result, crowding_metadata = apply_background_crowding(result, area)

    print("Standardizing columns.")
    result = _standardize_columns(result)

    return result, crowding_metadata


def _standardize_columns(result):
    result = result.drop(
        columns=[
            "temperature_boring",
            "A_None",
            "In_Final_Phase",
            "vr_bc",
            "mul",
            "mub",
            "x",
            "y",
            "z",
            "U",
            "V",
            "W",
            "VR_LSR",
        ]
    ).rename(
        columns={
            "iMass": "mass_initial",
            "Mass": "mass",
            "Dist": "distance",
            "Gaia_G_EDR3": "gaia_dr3_g",
            "Gaia_BP_EDR3": "gaia_dr3_bp_true",
            "Gaia_RP_EDR3": "gaia_dr3_rp_true",
            "2MASS_J": "2mass_j",
            "2MASS_H": "2mass_h",
            "2MASS_Ks": "2mass_k",
            "N": "gaianir_n",
            "N_R": "gaianir_r",
            "N_J": "gaianir_j",
            "N_H": "gaianir_h",
            "N_K": "gaianir_k",
        }
    )

    return result


def _inflate_area(area, simulated_stars, minimum_stars):
    # If there are no stars at all: "guess" at multiplying area by 10
    if simulated_stars == 0:
        return area * 10

    # Otherwise, always at least double the area trialled, or at least scale
    # it based on the number of stars simulated last time
    return area * np.clip(minimum_stars / simulated_stars * 2, 2, np.inf)


def _assign_true_astrometry(result, coords):
    result["ra"] = coords.ra.to(u.deg).value
    result["dec"] = coords.dec.to(u.deg).value
    result["pmra_true"] = coords.pm_ra_cosdec.to(u.mas / u.yr).value
    result["pmdec_true"] = coords.pm_dec.to(u.mas / u.yr).value
    result["parallax_true"] = 1 / result["Dist"]


def _calculate_gaianir_astrometry(result):
    model_gaianir_l = AstrometryModelElectronBased(mission="GaiaNIR-L")
    (
        result["ra_error_gaianir-l"],
        result["dec_error_gaianir-l"],
        result["pmra_error_gaianir-l"],
        result["pmdec_error_gaianir-l"],
        result["parallax_error_gaianir-l"],
    ) = model_gaianir_l.predict(result["N"])

    model_gaianir_m = AstrometryModelElectronBased(mission="GaiaNIR-M")
    (
        result["ra_error_gaianir-m"],
        result["dec_error_gaianir-m"],
        result["pmra_error_gaianir-m"],
        result["pmdec_error_gaianir-m"],
        result["parallax_error_gaianir-m"],
    ) = model_gaianir_m.predict(result["N"])


def _calculate_gaia_astrometry(result):
    """Adds Gaia DR4 and Gaia DR5 uncertainties to the observation."""

    # Calculate strength of reddening
    result["Gaia_G_EDR3"] = result["Gaia_G_EDR3"] + AG(
        result["extinction"], result["temperature"]
    )
    good_stars = result["Gaia_G_EDR3"] < FAINTEST_GAIA_MAGNITUDE
    result["temperature_boring"] = 5000  # Little/no impact for Gaia, so ignored

    model_dr4 = AstrometryModel(mission="Gaia", years=5)
    (
        result.loc[good_stars, "ra_error_gaia_dr4"],
        result.loc[good_stars, "dec_error_gaia_dr4"],
        result.loc[good_stars, "pmra_error_gaia_dr4"],
        result.loc[good_stars, "pmdec_error_gaia_dr4"],
        result.loc[good_stars, "parallax_error_gaia_dr4"],
    ) = model_dr4.predict(
        result.loc[good_stars],
        temperature_column="temperature_boring",
        mag_column="Gaia_G_EDR3",
    )

    model_dr5 = AstrometryModel(mission="Gaia")
    (
        result.loc[good_stars, "ra_error_gaia_dr5"],
        result.loc[good_stars, "dec_error_gaia_dr5"],
        result.loc[good_stars, "pmra_error_gaia_dr5"],
        result.loc[good_stars, "pmdec_error_gaia_dr5"],
        result.loc[good_stars, "parallax_error_gaia_dr5"],
    ) = model_dr5.predict(
        result.loc[good_stars],
        temperature_column="temperature_boring",
        mag_column="Gaia_G_EDR3",
    )


def _calculate_combined_astrometry(result):
    gaia_length = 10
    gaianir_length = 10

    good_stars = result["Gaia_G_EDR3"] < FAINTEST_GAIA_MAGNITUDE_USED

    result["pmra_error_gaianir-l_combined"] = result["pmra_error_gaianir-l"]
    result["pmdec_error_gaianir-l_combined"] = result["pmdec_error_gaianir-l"]
    (
        result.loc[good_stars, "pmra_error_gaianir-l_combined"],
        result.loc[good_stars, "pmdec_error_gaianir-l_combined"],
    ) = combine_astrometry(
        result.loc[good_stars, "ra_error_gaianir-l"],
        result.loc[good_stars, "dec_error_gaianir-l"],
        result.loc[good_stars, "ra_error_gaia_dr5"],
        result.loc[good_stars, "dec_error_gaia_dr5"],
        GAIA_GAIANIR_SEPARATION + gaianir_length / 2 + gaia_length / 2,
    )

    result["pmra_error_gaianir-m_combined"] = result["pmra_error_gaianir-m"]
    result["pmdec_error_gaianir-m_combined"] = result["pmdec_error_gaianir-m"]
    (
        result.loc[good_stars, "pmra_error_gaianir-m_combined"],
        result.loc[good_stars, "pmdec_error_gaianir-m_combined"],
    ) = combine_astrometry(
        result.loc[good_stars, "ra_error_gaianir-m"],
        result.loc[good_stars, "dec_error_gaianir-m"],
        result.loc[good_stars, "ra_error_gaia_dr5"],
        result.loc[good_stars, "dec_error_gaia_dr5"],
        GAIA_GAIANIR_SEPARATION + gaianir_length / 2 + gaia_length / 2,
    )


def _sample_astrometry(result):
    rng = np.random.default_rng()

    missions = [
        "gaianir-l",
        "gaianir-m",
        "gaia_dr4",
        "gaia_dr5",
        "gaianir-l_combined",
        "gaianir-m_combined",
    ]

    for mission in missions:
        result[f"pmra_{mission}"] = rng.normal(
            loc=result["pmra_true"], scale=result[f"pmra_error_{mission}"]
        )
        result[f"pmdec_{mission}"] = rng.normal(
            loc=result["pmdec_true"], scale=result[f"pmdec_error_{mission}"]
        )
        if "combined" not in mission:
            result[f"parallax_{mission}"] = rng.normal(
                loc=result["parallax_true"], scale=result[f"parallax_error_{mission}"]
            )

    return result


def _calculate_photometry(result, coords):
    result["label"] = 0
    result["luminosity"] = 10 ** result["logL"]
    result["temperature"] = 10 ** result["logTeff"]
    result["log_g"] = result["logg"]

    result = PHOTOMETRY_PREDICTOR.predict(
        result, result["Fe/H_evolved"], luminosity_in_L_sun=True
    )

    dist = 5 * np.log10(result["Dist"] * 1000) - 5
    for band in PHOTOMETRY_PREDICTOR.photometric_bands:
        result[band] = result[band] + dist

    _assign_extinctions(result, coords)
    return result


def _assign_extinctions(result, coords):
    # Grab extinction
    result["extinction_green"] = _bayestar_map.query(coords, mode="best")
    result["extinction_zucker"] = _zucker_map.query(coords, mode="mean")

    result["extinction"] = np.where(
        result["extinction_zucker"].notna(),
        result["extinction_zucker"],
        result["extinction_green"],
    )

    # Apply extinctions
    query_data = pd.DataFrame.from_dict(
        {
            "A0": np.clip(
                result["extinction"], 0, 50
            ),  # Clip as relation only valid to A_V=50
            "R0": 3.1,
            "teff": result["temperature"],
        }
    )

    for band in PHOTOMETRY_PREDICTOR.photometric_bands:
        result[band] = result[band] + result["extinction"] * np.asarray(
            EXTINCTION_MODELS[band].predict(query_data)
        )


def _get_skycoord(result):
    coords = SkyCoord(
        CartesianRepresentation(
            result["x"].to_numpy(),
            result["y"].to_numpy(),
            result["z"].to_numpy(),
            unit=u.kpc,
            differentials=CartesianDifferential(
                result["U"].to_numpy(),
                result["V"].to_numpy(),
                result["W"].to_numpy(),
                unit=u.km / u.s,
            ),
        ),
        frame="galactocentric",
    ).transform_to("icrs")

    return coords
