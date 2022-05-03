import xarray as xr
import os
import shutil

import pingrid


datadir = "/data/aaron/fbf-candidate"
base = "http://iridl.ldeo.columbia.edu"

datasets = [
    (
        "enso",
        "/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/zlev/removeGRID/X/-170/-120/RANGE/Y/-5/5/RANGEEDGES/dup/T/12.0/splitstreamgrid/dup/T2/(1856)/last/RANGE/T2/30.0/12.0/mul/runningAverage/T2/12.0/5.0/mul/STEP/%5BT2%5DregridLB/nip/T2/12/pad1/T/unsplitstreamgrid/sub/%7BY/cosd%7D%5BX/Y%5Dweighted-average/T/3/1.0/runningAverage/%7BLaNina/-0.45/Neutral/0.45/ElNino%7Dclassify/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/5/flagge/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/1.0/flagge/dup/a%3A/sst/(LaNina)/VALUE/%3Aa%3A/sst/(ElNino)/VALUE/%3Aa/add/1/maskge/dataflag/1/index/2/flagge/add/sst/(phil)/unitmatrix/sst_out/(Neutral)/VALUE/mul/exch/sst/(phil2)/unitmatrix/sst_out/(LaNina)/(ElNino)/VALUES/%5Bsst_out%5Dsum/mul/add/%5Bsst%5Ddominant_class//long_name/(ENSO%20Phase)/def/startcolormap/DATA/1/3/RANGE/blue/blue/blue/grey/red/red/endcolormap/T/(1980)/last/RANGE/data.nc"
    ),
    (
        "rain-malawi",
        "/SOURCES/.NOAA/.NCEP/.CPC/.Merged_Analysis/.monthly/.latest/.ver2/.prcp_est/X/32/36/RANGE/Y/-17/-9/RANGE/T/(Dec-Feb)/seasonalAverage/data.nc",
    ),
    (
        "pnep-malawi",
        "/home/.remic/.IRI/.FD/.NMME_Seasonal_HFcast_Combined/.malawi/.nonexceed/.prob/data.nc",
    ),
    (
        "rain-madagascar",
        "/home/.rijaf/.Madagascar_v3/.ALL/.monthly/.rainfall/.rfe/T/(Dec-Feb)/seasonalAverage/data.nc",
    ),
    (
        "pnep-madagascar",
        "/home/.aaron/.DGM/.Forecast/.Seasonal/.NextGen/.Madagascar_South/.PRCP/.pne/S/(1%20Sep)/(1%20Oct)/(1%20Nov)/VALUES/L/removeGRID/data.nc",
    ),
    (
        "rain-ethiopia",
        "/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/39.875/47.875/RANGE/Y/3.625/10.875/RANGE/T/(Mar-May)/seasonalAverage/30/mul//units/(mm/month)/def/data.nc",
    ),
    (
        "spi-ethiopia",
        "/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/39.875/47.875/RANGE/Y/3.625/10.875/RANGE/T/(Mar-May)/VALUES/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Mar-May)/VALUES/data.nc",
    ),
    (
        "ndvi-ethiopia",
        "/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/39.875/.25/47.875/GRID/Y/3.625/.25/10.875/GRID/T/(Mar-May)/seasonalAverage/data.nc",
    ),
    (
        "pnep-ethiopia",
        "/home/.aaron/.Ethiopia/.CPT/.NextGen/.MAM_PRCP/.Somali/.NextGen/.FbF/.pne/P//P//percentile/0/5/5/95/NewEvenGRID/replaceGRID/data.nc",
    ),
    (
        "rain-ethiopia-ond",
        "/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/30/mul/X/39.875/47.875/RANGE/Y/3.625/10.875/RANGE/T/(Oct-Dec)/seasonalAverage//units/(mm/month)/def/data.nc",
    ),
    (
        "pnep-ethiopia-ond",
        "/home/.aaron/.Ethiopia/.CPT/.NextGen/.OND_PRCP/.Somali/.NextGen/.FbF/.pne/P//P//percentile/0/5/5/95/NewEvenGRID/replaceGRID/data.nc",
    ),
    (
        'rain-niger',
        '/home/.aaron/.Niger/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JAS/.NextGen/.FbF/.obs/data.nc'
    ),
    (
        'pnep-niger',
        '/home/.aaron/.Niger/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JAS_v2/.NextGen/.FbF/.pne/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/data.nc'
    ),
    (
        'rain-guatemala',
        '/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Oct-Dec)/seasonalAverage//name//prcp_est/def/data.nc',
    ),
    (
        'pnep-guatemala',
        '/home/.xchourio/.ACToday/.CPT/.NextGen/.Seasonal/.CHIRPS/.GTM-FbF/.NextGen/.FbF/.pne/S/%281%20Sep%29VALUES/P/grid://name//P/def//units//percent/def/5/5/95/:grid/replaceGRID/L/removeGRID/data.nc'
    ),
    (
        'rain-djibouti',
        '/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Jul-Sep)/seasonalAverage/30/mul//units/(mm/month)/def/data.nc'
    ),
    (
        'pnep-djibouti',
        '/home/.remic/.ICPAC/.Forecasts/.CPT/.Djibouti/.prcp_JAS/.NextGen/.FbF/.pne/data.nc',
    )
]

for name, urlpath in datasets:
    print(name)
    ncfilepath = "%s/%s.nc" % (datadir, name)
    if os.path.exists(ncfilepath):
        timeopt = "--time-cond %s" % ncfilepath
    else:
        timeopt = ""
    os.system(
        "curl %s -o %s 'http://iridl.ldeo.columbia.edu%s'" % (timeopt, ncfilepath, urlpath)
    )
    assert os.path.exists(ncfilepath)
    zarrpath = "%s/%s.zarr" % (datadir, name)
    if (os.path.exists(zarrpath) and
        os.path.getctime(zarrpath) >= os.path.getctime(ncfilepath)):
        print("Zarr already exists")
    else:
        print("Converting to zarr")
        ds = pingrid.open_dataset(ncfilepath)
        # TODO do this in Ingrid
        if 'Y' in ds and ds['Y'][0] > ds['Y'][1]:
            ds = ds.reindex(Y=ds['Y'][::-1])
        if os.path.exists(zarrpath):
            shutil.rmtree(zarrpath)
        ds.to_zarr(zarrpath)
