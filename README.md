# The detectability of open clusters in GaiaNIR

This repo contains the source code for a project to determine the star cluster selection function of the proposed GaiaNIR astrometric mission. GaiaNIR would revolutionize Galactic astronomy with an incredibly high-accuracy and deep astrometric survey of the Milky Way, and is currently one of two options for the 8th L-class mission of the European Space Agency. This project wants to add more evidence about why it will be funded!

This repository and all source code is **public** for outreach & collaborative purposes. Please don't scoop the work here before then -- that would be pointless and rude >:(


## Installation

Use of [uv](https://docs.astral.sh/uv/) to install project dependencies is **strongly recommended**. After installing uv (which is [easy](https://docs.astral.sh/uv/getting-started/installation/)!), run

```bash
uv sync
```

in this directory to install all dependencies.

To use the code in practice, you will also need quite a lot of bits of data (atmosphere models, isochrones, and more). These can be downloaded automatically with the script `fetch_data.py`, which can be ran with

```bash
uv run fetch_data.py
```
