# Seasonal Forecast Maproom

This directory contains a Dash creation to visualize
CPT sub-Seasonal Forecasts (full distribution)


# Installation and Run Instructions

* create environment with

    `conda env create -f environment.yml`

* After having activated the environment, run application on devi as follows, after adapting a config.yaml from config-sample.yaml:

    `CONFIG=config.yaml python maproom.py`

* Navigate your browser to `http://devi:8063/subseas-flex-fcst-maproom/` (I am using 8063, please don't!)

* When done using the maproom stop Dash with CTRL-C and deactivate the environment with:

    `conda deactivate`

# Development Instructions

This maproom is structured around two different files:

* `layout.py`: functions which generate the general layout of the maproom

* `maproom.py`: callbacks for user interaction

# Docker Build Instructions

To build the docker image, we have to use a work around so that pingrid.py will be included correctly, as
docker doesn't normally allow files above the working directory in the hierarchy to be included

    $ tar -czh . | sudo docker build -t <desired image name> -

For final releases of the image, use the `release_container_image` script (no parameters) in this directory
to build and push to dockerhub.


# Support

* `help@iri.columbia.edu`
