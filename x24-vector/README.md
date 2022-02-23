# X24-vector

Dash application x24-vector overlays results of WRSI DL [query](http://iridl.ldeo.columbia.edu/home/.remic/.Leap/.WRSI/.Meher/.FinalIcat/Crop/%28Barley%29/VALUE/X/Y/fig-/colors/-fig) on the street or topo map and also displays points of interest from a CSV files that has at least the following columns:

* Region
* Woreda
* Kebele
* Lon
* Lat
* Primary Crops
* Expansion or Current

Additionally it uses a python library to convert white background into transparent background on the fly.

The image overlay is not normally recommended, the better (and a bit more involved) way to overlay raster data is tiling.
Please consult us if you require to overlay raster data.

# Screenshot

![Screenshot](x24.0-vector.png)

# Installation and Run Instructions

* Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

* Create conda environment `dash` as follows: 

    `conda env create -f environment.yml`

* Activate conda environment `dash` as follows:

    `conda activate dash`

* Run application as follows:

    `python x24-vector.py data.csv`

* Navigate your browser to http://127.0.0.1:8050/x24-vector/


