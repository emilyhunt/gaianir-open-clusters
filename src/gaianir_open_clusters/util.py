import os
import sys
import numpy as np
from gaianir_open_clusters.gaia_nir_config import POTENTIAL
from astropy.coordinates import (
    SkyCoord,
    CylindricalRepresentation,
    CylindricalDifferential,
)
from astropy import units as u


class HiddenPrints:
    """From https://stackoverflow.com/a/45669280"""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_circular_orbit_skycoord(l, b, distance, v_rho=0, v_z=0, v_phi_offset=0):
    """Fetches a SkyCoord at the given l,b,distance with a cluster on a perfect circular
    orbit, based on the chosen MW potential for the project.
    """
    # Fetch initial positional coords
    coords = SkyCoord(
        l=l * u.deg, b=b * u.deg, distance=distance * u.pc, frame="galactic"
    )
    pos_galactocentric = coords.transform_to("galactocentric")
    pos_cylindrical = pos_galactocentric.represent_as("cylindrical")

    # Calculate circular velocity
    v_circ = POTENTIAL.circular_velocity(
        [
            pos_galactocentric.x.to(u.kpc).value,
            pos_galactocentric.y.to(u.kpc).value,
            pos_galactocentric.z.to(u.kpc).value,
        ]
    )

    # Convert circular velocity to an angular velocity (rad/s) in a cylindrical frame
    v_phi_angular = (
        (-v_circ.to(u.km / u.s).value - v_phi_offset)
        / pos_cylindrical.rho.to(u.km).value
        * u.rad
        / u.s
    )

    # Fetch full coords object
    coords_full = SkyCoord(
        CylindricalRepresentation(
            rho=pos_cylindrical.rho,
            phi=pos_cylindrical.phi,
            z=pos_cylindrical.z,
            differentials=CylindricalDifferential(
                d_rho=np.full(len(l), v_rho) * u.km / u.s,
                d_phi=v_phi_angular,
                d_z=np.full(len(l), v_z) * u.km / u.s,
            ),
        ),
        frame="galactocentric",
    )
    return coords_full.transform_to("icrs")
