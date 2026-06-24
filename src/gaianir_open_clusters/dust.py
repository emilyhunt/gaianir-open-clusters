import numpy as np
from dustmaps.planck import PlanckQuery
from astropy.coordinates import galactocentric_frame_defaults
from astropy import units as u
from scipy.stats import truncexpon


DEFAULT_R_SUN = galactocentric_frame_defaults.get()["galcen_distance"].to(u.pc).value


class PlanckQuery3D:
    def __init__(
        self,
        max_distance=20000,
        scale_length=3000,
        expon_integral_resolution=100,
        r_sun=DEFAULT_R_SUN,
    ):
        """Interpolation of the Planck 2D dust map into 3D. Has two modes: exponential
        and linear. Exponential assumes a dusty disk with dust exponentially distributed
        (default scale length: 3 kpc), while linear just interpolates from a distance of
        zero to the edge of the galaxy. Exponential mode generally seems to work better.
        """
        self._dust_map = PlanckQuery()
        self.max_distance = max_distance
        self.expon_model = truncexpon(max_distance / scale_length, scale=scale_length)
        self.expon_integral_resolution = expon_integral_resolution
        self.r_sun = r_sun

    def query(
        self,
        coords,
        mode="expon",
        expon_integral_resolution=None,
        expon_return_full_distance_info: bool = False,
    ):
        """Returns dust reddening E(B-V) at coords."""
        coords = coords.transform_to("galactic")
        l_degrees = coords.l.to(u.deg).value
        distance = coords.distance.to(u.pc).value
        max_distances = self._calculate_maximum_distance(l_degrees)
        total_extinctions = self._dust_map.query(coords)

        if mode == "linear":
            distance_fractions = np.clip(distance / max_distances, 0, 1)
            return distance_fractions * total_extinctions

        elif mode == "expon":
            if expon_integral_resolution is None:
                expon_integral_resolution = self.expon_integral_resolution

            # Setup a grid of distances to evaluate the exponential model at
            distance_clipped = np.clip(distance, 0, max_distances)
            distance_integrals = np.linspace(
                np.zeros(len(max_distances)),
                max_distances,
                num=self.expon_integral_resolution,
            ).T

            # Convert the distances to galactocentric radius & calculate value
            rho_integrals = self._calculate_galactocentric_radius(
                l_degrees.reshape(-1, 1), distance_integrals
            )
            pdf_values = self.expon_model.pdf(rho_integrals)

            # Turn the PDF into a CDF along this direction
            cdf_values = np.cumsum(pdf_values, axis=1)
            cdf_values /= np.max(cdf_values, axis=1).reshape(-1, 1)

            # Optionally, return samples at all calculated distances, useful for
            # later interpolation or debugging
            if expon_return_full_distance_info:
                return cdf_values * total_extinctions, distance_integrals

            # Otherwise, interpolate at each coord and calculate interpolated extinction
            extinction_fraction = np.asarray(
                [
                    np.interp(d, d_i, c_i)
                    for d, d_i, c_i in zip(
                        distance_clipped, distance_integrals, cdf_values
                    )
                ]
            )
            return extinction_fraction * total_extinctions

        else:
            raise ValueError(f"selected mode '{mode}' not recognized.")

    def _calculate_maximum_distance(self, l_degrees):
        r_sun_cos_l = 2 * self.r_sun * np.cos(np.radians(l_degrees))
        return (
            r_sun_cos_l
            + np.sqrt(r_sun_cos_l**2 - 4 * (self.r_sun**2 - self.max_distance**2))
        ) / 2

    def _calculate_galactocentric_radius(self, l_degrees, distance):
        return np.sqrt(
            distance**2
            + self.r_sun**2
            - 2 * distance * self.r_sun * np.cos(np.radians(l_degrees))
        )
