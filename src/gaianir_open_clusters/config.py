import gaianir_open_clusters
import os
from pathlib import Path


INSTALLATION_DIRECTORY = Path(gaianir_open_clusters.__path__[0]).parent.parent
DATA_DIRECTORY = INSTALLATION_DIRECTORY / "data"
SYNTHPOP_CONFIG_DIRECTORY = INSTALLATION_DIRECTORY / "synthpop_configs"
RESULTS_DIRECTORY = INSTALLATION_DIRECTORY / "results"


# Synthpop things
SYNTHPOP_GAIANIR_CONFIG = str(SYNTHPOP_CONFIG_DIRECTORY / "gaia_nir.synthpop_conf")
SYNTHPOP_DEFAULT_CONFIG = str(
    SYNTHPOP_CONFIG_DIRECTORY / "huston2025_defaults.synthpop_conf"
)

# ocelot things
os.environ["OCELOT_DATA"] = (DATA_DIRECTORY / "ocelot_data").as_posix()
