# ENACTS Maprooms

# Installation and Run Instructions

## Creating a conda environment with this project's dependencies

* Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

* Create a conda environment named enactsmaproom from the enacts/ folder:

    ```
    conda create -n enactsmaproom --file conda-linux-64.lock
    ```
    (substituting osx or win for linux as appropriate)

    You don't need to install conda-lock for this.

    Note that the command is `conda create`, not `conda env create`. Both exist, and they're different :-(

## Running the application

* Activate the environment

    `conda activate enactsmaproom`

* Create a config.yaml  with your server specificities.

* Edit `config-<country>.yaml` as you see fit.

* Start the development server using both config files, e.g.:

    `CONFIG=../config.yaml:../config-sng.yaml:../myconfig.yaml python maproom_monit.py`

* Where `myconfig.yaml` contains any individual changes you would like applied to the other two config files.

* Navigate your browser to the URL that is displayed when the server starts, e.g. `http://127.0.0.1:8050/python_maproom/`

* When done using the server stop it with CTRL-C.

# Development Instructions

Maprooms are structured around four different files:

* `layout.py`: functions which generate the general layout of the maproom

* `maproom.py`: callbacks for user interaction

* `charts.py`: code for generating URLs for dlcharts/dlsnippets/ingrid charts and/or fetching table data

* `widgets.py`: routines for common maproom components.

The widgets module contains a few functions of note:

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


# Support

* `help@iri.columbia.edu`
