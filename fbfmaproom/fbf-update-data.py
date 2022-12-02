import argparse
import cftime
import xarray as xr
import os
import shutil

import pingrid

parser = argparse.ArgumentParser()
parser.add_argument('--cookiefile', type=os.path.expanduser)
parser.add_argument(
    '--datadir',
    default='/data/aaron/fbf-candidate',
    type=os.path.expanduser,
)
parser.add_argument('datasets', nargs='*')
opts = parser.parse_args()

base = "http://iridl.ldeo.columbia.edu"

url_datasets = [
    (
        "enso",
        "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.sst/zlev/removeGRID/X/-170/-120/RANGE/Y/-5/5/RANGEEDGES/dup/T/12.0/splitstreamgrid/dup/T2/(1856)/last/RANGE/T2/30.0/12.0/mul/runningAverage/T2/12.0/5.0/mul/STEP/%5BT2%5DregridLB/nip/T2/12/pad1/T/unsplitstreamgrid/sub/%7BY/cosd%7D%5BX/Y%5Dweighted-average/T/3/1.0/runningAverage/%7BLaNina/-0.45/Neutral/0.45/ElNino%7Dclassify/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/5/flagge/T/-2/1/2/shiftdata/%5BT_lag%5Dsum/1.0/flagge/dup/a%3A/sst/(LaNina)/VALUE/%3Aa%3A/sst/(ElNino)/VALUE/%3Aa/add/1/maskge/dataflag/1/index/2/flagge/add/sst/(phil)/unitmatrix/sst_out/(Neutral)/VALUE/mul/exch/sst/(phil2)/unitmatrix/sst_out/(LaNina)/(ElNino)/VALUES/%5Bsst_out%5Dsum/mul/add/%5Bsst%5Ddominant_class//long_name/(ENSO%20Phase)/def/startcolormap/DATA/1/3/RANGE/blue/blue/blue/grey/red/red/endcolormap/T/(1980)/last/RANGE/"
    ),
    (
        "rain-malawi",
        "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.Merged_Analysis/.monthly/.latest/.ver2/.prcp_est/X/32/36/RANGE/Y/-17/-9/RANGE/T/(Dec-Feb)/seasonalAverage/",
    ),
    (
        "pnep-malawi",
        "http://iridl.ldeo.columbia.edu/home/.remic/.IRI/.FD/.NMME_Seasonal_HFcast_Combined/.malawi/.nonexceed/.prob/",
    ),
    (
        "madagascar/enacts-precip-djf",
        "http://iridl.ldeo.columbia.edu/home/.rijaf/.Madagascar_v3/.ALL/.monthly/.rainfall/.rfe/T/(Dec-Feb)/seasonalAverage//units/(mm/month)/def/",
    ),
    (
        "madagascar/pnep-djf",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.DGM/.Forecast/.Seasonal/.NextGen/.Madagascar_Full/.DJF/.NextGen/.FbF/.pne/S/(0000%201%20Sep)/(0000%201%20Oct)/(0000%201%20Nov)/VALUES/",
    ),
    (
        "madagascar/enacts-precip-ond",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.ALL/.monthly/.rainfall/.rfe/T/(Oct-Dec)/seasonalAverage//units/(mm/month)/def/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/",
    ),
    (
        "madagascar/enacts-precip-mon-ond",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.MON/.monthly/.rainfall/.rfe/T/(Oct-Dec)/seasonalAverage//units/(mm/month)/def/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/",
    ),
    (
        "madagascar/chirps-precip-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/T/(Oct-Dec)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/",
    ),
    (
        "madagascar/ndvi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.SAF/.NDVI/X/42.525/.0375/48.975/GRID/Y/-25.9875/.0375/-20.025/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "madagascar/wrsi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.EROS/.FEWS/.dekadal/.SAF/.do/X/42.525/48.975/RANGE/Y/-25.9875/-20.025/RANGE/T/(Oct-Dec)/seasonalAverage/T/(months%20since%201960-01-01)/streamgridunitconvert/",
    ),
    (
        "madagascar/enacts-mon-spi-ond",
        "http://map.meteomadagascar.mg/SOURCES/.Madagascar_v4/.MON/.seasonal/.rainfall/.SPI-3-month/.spi/T/(Oct-Dec)/VALUES/",
    ),
    (
        "madagascar/pnep-ond",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.DGM/.Forecast/.Seasonal/.NextGen/.Madagascar_Full/.OND/.NextGen/.FbF/.pne/S/(1%20Jul)/(1%20Aug)/(1%20Sep)/VALUES/",
    ),
    (
        "ethiopia/rain-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Mar-May)/seasonalAverage/30/mul//units/(mm/month)/def/",
    ),
    (
        "ethiopia/rain-prev-seas-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Oct-Dec)/seasonalAverage/30/mul//units/(mm/month)/def/T/5/shiftGRID/",
    ),
    (
        "ethiopia/rain-prev-seas-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Mar-May)/seasonalAverage/30/mul//units/(mm/month)/def/T/7/shiftGRID/",
    ),
    (
        "ethiopia/spi-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Mar-May)/VALUES/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Mar-May%201981)/last/12/RANGESTEP/"
    ),
    (
        "ethiopia/spi-prev-seas-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Mar-May)/VALUES/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Mar-May%201981)/last/12/RANGESTEP/T/7/shiftGRID/",
    ),
    (
        "ethiopia/spi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Oct-Dec)/VALUES/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Oct-Dec%201981)/last/12/RANGESTEP/",
    ),
    (
        "ethiopia/spi-prev-seas-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Oct-Dec)/VALUES/monthlyAverage/30/mul/a%3A/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7D/def/-1./shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/%3Aa%3A/T/3/runningAverage/T/12/splitstreamgrid/0./flaggt/%5BT2%5Daverage/1./3./div/flaggt/1./masklt/%5BT%5D/REORDER/CopyStream/%3Aa/mul/DATA/-3/3/RANGE//name//spi/def/T/(Oct-Dec%201981)/last/12/RANGESTEP/T/5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Mar-May)/seasonalAverage/",
    ),
    (
        "ethiopia/ndvi-jun-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jun)/seasonalAverage/T/5/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-jul-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Jul)/seasonalAverage/T/4/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-aug-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Aug)/seasonalAverage/T/3/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Oct-Dec)/seasonalAverage/",
    ),
    (
        "ethiopia/ndvi-prev-seas-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Mar-May)/seasonalAverage/T/7/shiftGRID/",
    ),
    (
        "ethiopia/ndvi-prev-seas-mam",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/32.625/.25/48.375/GRID/Y/2.625/.25/15.375/GRID/T/(Oct-Dec)/seasonalAverage/T/5/shiftGRID/",
    ),
    (
        "ethiopia/pnep-mam",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.Ethiopia/.CPT/.NextGen/.MAM_PRCP/.Ethiopia/.NextGen/.FbF/.pne/P//P//percentile/0/5/5/95/NewEvenGRID/replaceGRID/",
    ),
    (
        "ethiopia/rain-ond",
        "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/30/mul/X/32.625/48.375/RANGE/Y/2.625/15.375/RANGE/T/(Oct-Dec)/seasonalAverage//units/(mm/month)/def/",
    ),
    (
        "ethiopia/pnep-ond",
        "http://iridl.ldeo.columbia.edu/home/.aaron/.Ethiopia/.CPT/.NextGen/.OND_PRCP/.Ethiopia/.NextGen/.FbF/.pne/P//P//percentile/0/5/5/95/NewEvenGRID/replaceGRID/"
    ),
    (
        'niger/enacts-precip-jas',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.ENACTS/.monthly/.rainfall/.CHIRPS/.rfe_merged/T/(1991)/last/RANGE/T/(Jul-Sep)/seasonalAverage/3/mul///name//obs/def/'
    ),
    (
        'niger/chirps-precip-jun',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(0)/(16)/RANGE/Y/(11)/(24)/RANGE/T/(Jun)/seasonalAverage/T//pointwidth/0/def/2/shiftGRID/c%3A/1//units//months/def/%3Ac/mul//name//precipitation/def/'
    ),
    (
        'niger/chirps-precip-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(0)/(16)/RANGE/Y/(11)/(24)/RANGE/T/(Jul-Sep)/seasonalAverage/c%3A/3//units//months/def/%3Ac/mul//name//precipitation/def/'
    ),
    (
        'niger/chirps-precip-jjaso',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/(0)/(16)/RANGE/Y/(11)/(24)/RANGE/T/(Jun-Oct)/seasonalAverage/c%3A/5//units//months/def/%3Ac/mul/T//pointwidth/0/def/pop//name//precipitation/def/'
    ),
    (
        'niger/enacts-spi-jas',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.ENACTS/.seasonal/.rainfall/.CHIRPS/.SPI-3-month/.spi/T/(Jul-Sep)/VALUES/',
    ),
    (
        'niger/enacts-spi-jj',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Niger/.ENACTS/.seasonal/.rainfall/.CHIRPS/.SPI-2-month/.spi/T/(Jun-Jul)/VALUES/T//pointwidth/0/def/1.5/shiftGRID/',
    ),
    (
        'niger/chirp-spi-jj',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/0/15.975/RANGE/Y/11.025/23.975/RANGE/monthlyAverage/3./mul/2/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/2/gammaprobs/2/gammastandardize/T//pointwidth/2/def//defaultvalue/%7Blast%7Ddef/-0.5/shiftGRID/T/first/pointwidth/1/sub/add/last/RANGE//long_name/(Standardized%20Precipitation%20Index)/def/DATA/-3/3/RANGE/T/(Jun-Jul)/VALUES/T//pointwidth/0/def/1.5/shiftGRID/',
    ),
    (
        'niger/chirp-spi-jas',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRP/.v1p0/.dekad/.prcp/X/0/15.975/RANGE/Y/11.025/23.975/RANGE/monthlyAverage/3.0/mul/3/gamma3par/pcpn_accum/gmean/gsd/gskew/pzero/3/gammaprobs/3/gammastandardize/T//pointwidth/3/def//defaultvalue/%7Blast%7Ddef/(Jul-Sep)/VALUES//long_name/(Standardized%20Precipitation%20Index)/def/DATA/-3/3/RANGE/',
    ),
    (
        'niger/chirps-dryspell',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp/X/(0)/(16)/RANGE/Y/(11)/(24)/RANGE/T/(1%20Jun)/61/1/(lt)/0.9/seasonalLLS/T//pointwidth/0/def/(months%20since%201960-01-01)/streamgridunitconvert/T/1.5/shiftGRID/'
    ),
    (
        'niger/chirps-onset',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp/X/(0)/(16)/RANGE/Y/(11)/(24)/RANGE/T/({start})/({end})/RANGE/T/(1%20May)/122/0.1/3/20/1/15/30/onsetDate/T/sub/T//pointwidth/0/def/(months%20since%201960-01-01)/streamgridunitconvert/T/3.5/shiftGRID/',
        (
            {'start': 1991, 'end': 2000},
            {'start': 2001, 'end': 2010},
            {'start': 2011, 'end': 2020},
            {'start': 2021, 'end': 2022},
        )
    ),
    (
        'niger/pnep-jas-v3',
        'http://iridl.ldeo.columbia.edu/home/.remic/.Niger/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JAS_v3/.NextGen/.FbF/.pne/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/'
    ),
    (
        'niger/pnep-jj',
        'http://iridl.ldeo.columbia.edu/home/.remic/.Niger/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JJ/.NextGen/.FbF/.pne/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/'
    ),
    (
        'niger/poe-onset',
        'http://iridl.ldeo.columbia.edu/home/.remic/.Niger/.Forecasts/.NextGen/.PRCPonset_CCAFCST_{part}/.NextGen/.FbF/.pne/100/exch/sub//name//poe/def//long_name/(Probability%2520of%2520exceedance)/def/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/',
        (
            {'part': 'N'},
            {'part': 'S'},
        ),
    ),
    (
        'niger/poe-dryspell',
        'http://iridl.ldeo.columbia.edu/home/.remic/.Niger/.Forecasts/.NextGen/.PRCPmaxDS_CCAFCST_JA/.NextGen/.FbF/.pne/100/exch/sub//name//poe/def//long_name/(Probability%20of%20exceedance)/def/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/'
    ),
    (
        'niger/enacts-mon-spi-jj',
        'http://iridl.ldeo.columbia.edu/home/.rijaf/.Niger/.ENACTS/.MON/.seasonal/.rainfall/.CHIRP/.SPI-2-month/.spi/T/%28Jun-Jul%29VALUES/T/1.5/shiftGRID/',
    ),
    (
        'niger/enacts-mon-spi-jas',
        'http://iridl.ldeo.columbia.edu/home/.rijaf/.Niger/.ENACTS/.MON/.seasonal/.rainfall/.CHIRP/.SPI-3-month/.spi/T/%28Jul-Sep%29VALUES/',
    ),
    (
        'rain-guatemala',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/X/-92.5/.1/-88/GRID/Y/13/.1/18/GRID/T/(Oct-Dec)/seasonalAverage//name//prcp_est/def/',
    ),
    (
        'pnep-guatemala',
        'http://iridl.ldeo.columbia.edu/home/.xchourio/.ACToday/.CPT/.NextGen/.Seasonal/.CHIRPS/.GTM-FbF/.NextGen/.FbF/.pne/S/%281%20Sep%29VALUES/P/grid://name//P/def//units//percent/def/5/5/95/:grid/replaceGRID/L/removeGRID/'
    ),
    (
        'rain-djibouti',
        'http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/X/41.625/43.375/RANGE/Y/10.875/12.875/RANGE/T/(Jul-Sep)/seasonalAverage/30/mul//units/(mm/month)/def/'
    ),
    (
        'djibouti/pnep-jas',
        'http://iridl.ldeo.columbia.edu/home/.aaron/.Djibouti/.PRCPPRCP_CCAFCST/.NextGen/.FbF/.pne/',
    ),
    (
        "ndvi-djibouti",
        "http://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI/X/41.625/.25/43.375/GRID/Y/10.875/.25/12.875/GRID/T/(Jul-Sep)/seasonalAverage/",
    ),
    (
        "lesotho/pnep-djf",
        "http://iridl.ldeo.columbia.edu/home/.remic/.Lesotho/.Forecasts/.NextGen/.DJF_PRCPPRCP_CCAFCST/.NextGen/.FbF/.pne/",
    ),
    (
        "lesotho/enacts-precip-djf",
        "https://iridl.ldeo.columbia.edu/home/.audreyv/.dle_lms/.Lesotho/.ENACTS/.ALL/.monthly/.rainfall/.rfe/T/(Dec-Feb)/seasonalAverage/3/mul/",
    ),
    (
        "lesotho/pnep-ond",
        "http://iridl.ldeo.columbia.edu/home/.remic/.Lesotho/.Forecasts/.NextGen/.OND_PRCPPRCP_CCAFCST/.NextGen/.FbF/.pne/",
    ),
    (
        "lesotho/enacts-precip-ond",
        "https://iridl.ldeo.columbia.edu/home/.audreyv/.dle_lms/.Lesotho/.ENACTS/.ALL/.monthly/.rainfall/.rfe/T/(Oct-Dec)/seasonalAverage/3/mul/",
    ),
]


selected_url_datasets = opts.datasets or [ds[0] for ds in url_datasets]

for dataset in url_datasets:
    name = dataset[0]
    pattern = dataset[1]
    if len(dataset) == 3:
        slices = dataset[2]
    else:
        slices = ({},)

    if name not in selected_url_datasets:
        continue
    print(name)
    for i, args in enumerate(slices):
        ncfilepath = f'{opts.datadir}/{name}-{i}.nc'
        leafdir = os.path.dirname(ncfilepath)

        if not os.path.exists(leafdir):
            os.makedirs(leafdir)
        if os.path.exists(ncfilepath):
            timeopt = "--time-cond %s" % ncfilepath
        else:
            timeopt = ""

        if opts.cookiefile is None:
            cookieopt = ""
        else:
            cookieopt = f"-b {opts.cookiefile}"

        url = pattern.format(**args)
        os.system(f"curl {timeopt} {cookieopt} -o {ncfilepath} '{url}data.nc'")
        assert os.path.exists(ncfilepath)
    zarrpath = "%s/%s.zarr" % (opts.datadir, name)
    if (os.path.exists(zarrpath) and
        os.path.getctime(zarrpath) >= os.path.getctime(ncfilepath)):
        print("Zarr already exists")
    else:
        print("Converting to zarr")
        ds = pingrid.open_mfdataset([f'{opts.datadir}/{name}-{i}.nc' for i in range(len(slices))])
        # TODO do this in Ingrid
        if 'Y' in ds and ds['Y'][0] > ds['Y'][1]:
            ds = ds.reindex(Y=ds['Y'][::-1])
        if 'P' in ds:
            ds = ds.chunk({'P': 1})
        if os.path.exists(zarrpath):
            shutil.rmtree(zarrpath)
        ds.to_zarr(zarrpath)
