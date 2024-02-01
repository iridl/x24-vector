# ENACTS Maprooms

# Installation and Run Instructions

## Creating a conda environment with this project's dependencies

* Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

* Create a conda environment named enactsmaproom:

    ```
    conda create -n enactsmaproom --file conda-linux-64.lock
    ```
    (substituting osx or win for linux as appropriate)

    You don't need to install conda-lock for this.

    Note that the command is `conda create`, not `conda env create`. Both exist, and they're different :-(

## Running the application in a development environment

* Activate the environment

    `conda activate enactsmaproom`

* Edit or use config.yaml as an example to create a config file with your server specificities.

* Edit `config-<country>.yaml` as you see fit. To exclude one of the maprooms, set its configuration to `null`, e.g.

    ```
    maprooms:
        flex_fcst: null
    ```

* Start the development server using both config files, e.g.:

    `CONFIG=config.yaml:config-zmd.yaml python app.py`

* Navigate your browser to the URL that is displayed when the server starts, e.g. `http://127.0.0.1:8050/python_maproom/`

* When done using the server stop it with CTRL-C.

# Development Overview

Maprooms are structured around four different files:

* `layout.py`: functions which generate the general layout of the maproom

* `maproom.py`: callbacks for user interaction

* `charts.py`: code for generating URLs for dlcharts/dlsnippets/ingrid charts and/or fetching table data

* `controls.py`: routines for common maproom controls.

The controls module contains a few functions of note:

* `Body()`: The first parameter is a string which is the title of the layout block.
   After the first, this function allows for a variable number of Dash components.

* `Sentence()`: Many maprooms have forms in a "sentence" structure where input fields are interspersed
  within a natural language sentence. This function abstracts this functionality. It requires that
  the variable number of arguments alternate between a string and a dash component.

* `Date()`: This is a component for a date selector. The first argument is the HTML id,
  the second is the default day of month, and the third is the default month (in three-letter abbreviated form)

* `Number()`: This is a component for a number selector. The first argument is the HTML id,
   the second and third are the lower and upper bound respectively.


## Adding or removing dependencies

Edit `environment.yml`, then regenerate the lock files as follows:
```
conda-lock lock -f environment.yml -f environment-dev.yml
conda-lock render
```



## Building the documentation

After creating and activating the conda environment (see above),

    make html

Then open (or reload) `build/html/index.html` in a browser.

The markup language used in docstrings is [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). Follow the [numpy Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html).


# Docker Build Instructions

To build the docker image, we have to use a work around so that pingrid.py will be included correctly, as
docker doesn't normally allow files above the working directory in the hierarchy to be included

    $ tar -czh . | sudo docker build -t <desired image name> -

For final releases of the image, use the `release_container_image` script (no parameters) in this directory
to build and push to dockerhub.


# Running enactstozarr on a partner DL

After having installed a Dockerized system, and after having creating appropriate folders in `/data/datalib/data/` according to the partner's configuration, run the command:

    sudo docker run \
      --rm \
      -u $(id -u) \
      -v /data/datalib/data:/data/datalib/data:rw \
      -v /usr/local/datalib/build/python_maproom/config.yaml:/app/config.yaml \
      -e CONFIG=/app/config.yaml \
      iridl/enactsmaproom \
      python enactstozarr.py

Should you need to run a modified version of the enactstozarr script, add the option:

      -v myenactstozarrpath/enactstozarr.py:/app/enactstozarr.py

where `myenactstozarrpath` is the path where you have your modified `enactstozarr.py`

# Support

* `help@iri.columbia.edu`
