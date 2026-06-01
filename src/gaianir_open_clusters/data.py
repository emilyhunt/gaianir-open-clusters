import pooch

# import tarfile
from gaianir_open_clusters.config import DATA_DIRECTORY
from pathlib import Path


def fetch_kurucz_models():
    """Ensures that Kurucz atmosphere models are downloaded and usable. Will only
    download the first time; future times will auto-skip."""
    pooch.retrieve(
        "https://keeper.mpdl.mpg.de/f/a80ede0816674d729f4e/?dl=1",
        known_hash="9576c2d2f0b8a78a6e1e569fe47dfc2615aa93dc279afe7b482ff8e9bf62c439",
        fname="Kurucz2003.tar.gz",
        path=DATA_DIRECTORY / "models",
        processor=pooch.Untar(
            extract_dir=DATA_DIRECTORY / "models"
        ),
    )

    # Remove ._ files from OSX (why tf do they get packaged with it!!!)
    bad_files = Path(DATA_DIRECTORY / "models").glob("**/._*")
    for file in bad_files:
        file.unlink()
