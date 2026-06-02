"""Fetches data for the project."""

from gaianir_open_clusters import data


print("Downloading kurucz atmosphere models")
data.fetch_kurucz_models()

print("Downloading ocelot data")
data.fetch_ocelot_data()

print("Downloading BT-Settl atmosphere models")
# data.fetch_ocelot_data()

print("Downloading GaiaNIR simulation results")
data.fetch_gaia_nir_astrometry()


print("Done!")