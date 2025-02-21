# ENACTS Maprooms

# Installation and Run Instructions

## One-time setup

* Install Pixi: https://pixi.sh/latest/

* Create a development configuration file by copying `config-dev-sample.yaml` to `config-dev.yaml` and editing it as needed. Note that `config-dev.yaml` is in the `.gitignore` file so you won't accidentally commit changes that are specific to your development environment.

## Running the application in a development environment

* cd to this project's directory

    `cd python-maprooms/enacts`

* Activate the development environment. The first time you run this, it will download and install the dependencies, which may take a few minutes. Subsequent times will be fast.

    `pixi shell`

Unlike conda, pixi runs the environment in a subshell. To "deactivate" the environment, exit the shell and you will be returned to your previous environment.

* Start the development server using both the country-specific config file and your development config file, e.g.:

    `CONFIG=../../python_maproom_mycountry/config.yaml:config-dev.yaml python app.py`

To test on an IRI server, using local copies of the data, add the country-specific IRI config file:

    `CONFIG=../../python_maproom_mycountry/config.yaml:../../python_maproom_mycountry/config-iri.yaml:config-dev.yaml python app.py`

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


## Adding, removing, or updating dependencies

```
pixi add package1
pixi remove package2
pixi add package3=1.0.1
pixi lock

git add pixi.toml pixi.lock
git commit ...
```


## Building the documentation

After creating and activating the python environment (see above),

    make html

Then open (or reload) `build/html/index.html` in a browser.

The markup language used in docstrings is [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). Follow the [numpy Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html).


# Docker Build Instructions

To build the docker image, we have to use a work around so that pingrid.py will be included correctly, as
docker doesn't normally allow files above the working directory in the hierarchy to be included

    tar czh --exclude-vcs-ignores . | sudo docker build -t iridl/enactsmaproom:latest -

For final releases of the image, build as above, then tag the image with the current date and push to Docker Hub:

    docker tag iridl/enactsmaproom:latest iridl/enactsmaproom:20240205
    docker login
    docker push iridl/enactsmaproom:latest
    docker push iridl/enactsmaproom:20240205
    docker logout

# Finding the most recent published container image
Visit https://hub.docker.com/r/iridl/enactsmaproom/tags . Clicking on the most recent tag will take you to a page that gives the sha256 hash of the image.

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


# Support

* `help@iri.columbia.edu`
