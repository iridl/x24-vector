# Monthly Climatology Maproom

To run this application, first download and install
[miniconda](https://docs.conda.io/en/latest/miniconda.html), then
build a python environment as follows:

```
# choose the lockfile that corresponds to the operating system you're using
> conda create --file conda-linux-64.lock -n  maproom-tutorial
```

To run the application,

```
> conda activate maproom-tutorial
> CONFIG=config-sample.yaml python maproom.py
```
then in a browser visit the URL that is printed in the output of the above command.
