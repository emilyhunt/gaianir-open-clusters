import pooch
from gaianir_open_clusters.config import DATA_DIRECTORY
from gaianir_open_clusters.gaia_nir_config import ASTROMETRIC_DATA_URL, ASTROMETRIC_DATA
from pathlib import Path


def fetch_kurucz_models():
    """Ensures that Kurucz atmosphere models are downloaded and usable. Will only
    download the first time; future times will auto-skip."""
    pooch.retrieve(
        "https://keeper.mpdl.mpg.de/f/a80ede0816674d729f4e/?dl=1",
        known_hash="9576c2d2f0b8a78a6e1e569fe47dfc2615aa93dc279afe7b482ff8e9bf62c439",
        fname="Kurucz2003.tar.gz",
        path=DATA_DIRECTORY / "models",
        processor=pooch.Untar(extract_dir=DATA_DIRECTORY / "models"),
    )

    # Remove ._ files from OSX (why tf do they get packaged with it!!!)
    bad_files = Path(DATA_DIRECTORY / "models").glob("**/._*")
    for file in bad_files:
        file.unlink()


def fetch_ocelot_data():
    """Fetches data used by ocelot."""
    pooch.retrieve(
        "https://cloud.emily.space/public.php/dav/files/AWmgtdMJQwByBEd",
        known_hash="344b7abb099d411455e55c11b0a07fc0c5956b9b4041be35780db675a820c2f9",
        fname="ocelot_data.zip",
        path=DATA_DIRECTORY,
        processor=pooch.Unzip(extract_dir=DATA_DIRECTORY / "ocelot_data"),
    )


def fetch_gaia_nir_astrometry():
    """Fetches GaiaNIR astrometric simulation results."""
    pooch.retrieve(
        ASTROMETRIC_DATA_URL,
        known_hash="a20b7f1768340d63cbff5463a7b2cd7948e41e5dbdaf993c929f7616f62070de",
        fname=ASTROMETRIC_DATA.stem + ".zip",
        path=DATA_DIRECTORY,
        processor=pooch.Unzip(extract_dir=ASTROMETRIC_DATA.parent),
    )
