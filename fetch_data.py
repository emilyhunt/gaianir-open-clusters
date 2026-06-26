"""Fetches data for the project."""

from gaianir_open_clusters import data
from dustmaps.bayestar import fetch as fetch_bayestar
from dustmaps.decaps import fetch as fetch_decaps
from dustmaps.planck import fetch as fetch_planck


print("Downloading kurucz atmosphere models")
data.fetch_kurucz_models()

# print("Downloading BT-Settl atmosphere models")
# data.fetch_btsettl_models()  # Todo: add a download link for them

print("Downloading ocelot data")
data.fetch_ocelot_data()

print("Downloading GaiaNIR simulation results")
data.fetch_gaia_nir_astrometry()

print(
    "Downloading dust maps (will not re-download if you already have them on your machine)"
)
fetch_bayestar()
fetch_decaps(mean_only=True, silence_warnings=True)
fetch_planck()


print("Done!")
