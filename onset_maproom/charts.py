import urllib.request
import json
import calc

PREFIX = "https://iridl.ldeo.columbia.edu/dlcharts/render/"

def _bounding_box(lat, lng):
    diff = 0.25
    lng_lo = lng - diff
    lng_hi = lng + diff
    lat_lo = lat - diff
    lat_hi = lat + diff
    return "bb%3A{0:.3f}%3A{1:.3f}%3A{2:.3f}%3A{3:.3f}%3Abb".format(lng_lo, lng_hi, lat_lo, lat_hi)

def _params(lat, lng, params):
    return "region=" + _bounding_box(lat, lng) + \
           "".join([ "&" + k + "=" + str(v) for k, v in params.items()])

def dlchart(chart, lat, lng, params):
    return PREFIX + chart + "?_wd=1200px&_ht=600px&_langs=en&_mimetype=image%2Fpng&" + \
           _params(lat, lng, params)

def onset_date(lat, lng, params):
    return dlchart("c4bb631e126f85cf413519e6d4ba2031dbe293b1", lat, lng, params)

def prob_exceed(lat, lng, params):
    return dlchart("cfa18aaf7a15fe79d690f5699ec0fd8b11538c2e", lat, lng, params)

def cess_date(lat, lng, params):
    return dlchart("e97365747da3eec3763a7e292ce5cf454728de99", lat, lng, params)

def cess_exceed(lat, lng, params):
    base = """http://213.55.84.78:8082/SOURCES/.Ethiopia/.NMA/.daily/.rainfall/.rfe_merged/X/33.0/0.25/48.0/GRID/Y/3.0/0.25/15.0/GRID/%28bb:38.8:9.2:38.9:9.3:bb%29//region/geoobject%5BX/Y%5Dweighted-average/SOURCES/.Ethiopia/.NMA/.monthly/.climatologies/.tmin//region/get_parameter/geoobject%5BX/Y%5Dweighted-average/0.0/mul/5.0/add%5BT%5DregridAverage/2/RECHUNK/%7Brainfall/ET%7Dds/dup/T/-1/shiftGRID/100.0/0.0/mul/rainfall/T/first/VALUE/T/removeGRID/0.0/mul/add/rainfall/.T/beginLoop/rainfall/add/ET/sub/60/min/0/max/endLoop/nip/dup/T/last/VALUE/T/1/shiftGRID/rainfall/add/ET/sub/60/min/0/max/appendstream//long_name/%28Soil%20Moisture%29def/T/1/index/T/%281%20Sep%29//earlyCess/parameter/VALUES/.T/.gridvalues/%7B90//searchDaysCess/parameter/3//drySpellCess/parameter/add/%7Bdup/1/add/dup/last/gt/%7Bpop%7Dif%7Drepeat%7Dforall/VALUES/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I//searchDaysCess/get_parameter//drySpellCess/get_parameter/add/1/add/splitstreamgrid/I//gridtype/0/def/pop/5//waterBalanceCess/parameter/flaglt/I/0/1/2/shiftdatashort%5BI_lag%5Dsum//drySpellCess/get_parameter/flagge/1/masklt/1/index/.T/0.0/add//name//datesample/def/T/npts//I/exch/NewIntegerGRID/replaceGRID/I//searchDaysCess/get_parameter//drySpellCess/get_parameter/add/1/add/splitstreamgrid/I//gridtype/0/def/pop/exch/mul%5BI%5Dminover/I2/3/-1/roll/T//earlyCess/get_parameter/VALUES/.T/2/array/astore/%7Bnpts%7Dforall/3/-1/roll/ne/%7Bfirst/secondtolast/subgrid%7Dif/dup//searchDaysCess/get_parameter/add/4/-3/roll/replaceGRID/exch/replaceNaN//pointwidth/1/def//long_name/%28Cessation%20Date%20since%20%29//earlyCess/get_parameter/append/def/T/sub/dup/DATA/0//plotrange1/parameter//searchDaysCess/get_parameter//plotrange2/parameter/1/RANGESTEP/integrateddistrib1D/-1/mul/1/add/%28/percent%29//poeunits/parameter/interp/unitconvert//poeunits/get_parameter/%28/unitless%29eq/%7B10.0/mul//units/%28years%20our%20of%2010%29def/DATA/0/10%7D%7BDATA/0/100%7Difelse/RANGE//long_name/%28probability%20of%20exceeding%29def/adif//long_name/%28Cessation%20Date%20since%20%29//earlyCess/get_parameter/append/def//cessationDate/renameGRID/a-/-a/cessationDate/fig-/blue/medium/line/-fig//framelabel%5B%281st%20d%20from%20%29//earlyCess/get_parameter/s==/%28%20in%20%29//searchDaysCess/get_parameter/s==/%28%20d%2C%20SWB%20of%20less%20than%20%29//waterBalanceCess/get_parameter/s==/%28%20mm%20for%20%29//drySpellCess/get_parameter/s==/%28%20d%29//region/get_parameter/geoobject/.long_name%5Dconcat/psdef/+.gif?"""
    return base + _params(lat, lng, params)

def pdf(lat, lng, params):
    return "https://iridl.ldeo.columbia.edu/dlsnippets/render/a1e11edea86bf533fb81bda1d97157abc546d943?_wd=1200px&_ht=600px&_languages=en&_mimetypes=image%2Fpng&" + \
           _params(lat, lng, params)

def table(lat, lng, params):
    onset_url = "http://213.55.84.78:8082/SOURCES/.Ethiopia/.NMA/.daily/.rainfall/.rfe_merged/X/33.0/0.25/48.0/GRID/Y/3.0/0.25/15.0/GRID/(bb%3A38.8%3A9.2%3A38.9%3A9.3%3Abb)//region/geoobject/%5BX/Y%5Dweighted-average/T/(1%20Jun)//earlyStart/parameter/90//searchDays/parameter/1//rainDay/parameter/5//runningDays/parameter/20//runningTotal/parameter/3//minRainyDays/parameter/7//dryDays/parameter/21//drySpell/parameter/onsetDate/T/dup/yearlyedgesgrid/first/secondtolast/subgrid/replaceGRID/T/(months%20since%201960-01-01)/streamgridunitconvert/T//pointwidth/12/def/6/shiftGRID/info.json?"
    cess_url = "http://213.55.84.78:8082/SOURCES/.Ethiopia/.NMA/.daily/.rainfall/.rfe_merged/X/33.0/0.25/48.0/GRID/Y/3.0/0.25/15.0/GRID/(bb%3A38.8%3A9.2%3A38.9%3A9.3%3Abb)//region/geoobject/%5BX/Y%5Dweighted-average/SOURCES/.Ethiopia/.NMA/.monthly/.climatologies/.tmin//region/get_parameter/geoobject/%5BX/Y%5Dweighted-average/0./mul/5./add/%5BT%5DregridAverage/2/RECHUNK/%7Brainfall/ET%7Dds/dup/T/-1/shiftGRID/100.0/0.0/mul/rainfall/T/first/VALUE/T/removeGRID/0.0/mul/add/rainfall/.T/beginLoop/rainfall/add/ET/sub/60/min/0/max/endLoop/nip/dup/T/last/VALUE/T/1/shiftGRID/rainfall/add/ET/sub/60/min/0/max/appendstream//long_name/(Soil%20Moisture)/def/T/1/index/T/(1%20Sep)//earlyCess/parameter/VALUES/.T/.gridvalues/%7B90//searchDaysCess/parameter/3//drySpellCess/parameter/add/%7Bdup/1/add/dup/last/gt/%7Bpop%7Dif%7Drepeat%7Dforall/VALUES/dup/T/npts//I/exch/NewIntegerGRID/replaceGRID/I//searchDaysCess/get_parameter//drySpellCess/get_parameter/add/1/add/splitstreamgrid/I//gridtype/0/def/pop/5//waterBalanceCess/parameter/flaglt/I/0/1/2/shiftdatashort/%5BI_lag%5Dsum//drySpellCess/get_parameter/flagge/1/masklt/1/index/.T/0.0/add//name//datesample/def/T/npts//I/exch/NewIntegerGRID/replaceGRID/I//searchDaysCess/get_parameter//drySpellCess/get_parameter/add/1/add/splitstreamgrid/I//gridtype/0/def/pop/exch/mul/%5BI%5Dminover/I2/3/-1/roll/T//earlyCess/get_parameter/VALUES/.T/2/array/astore/%7B/npts%7D/forall/3/-1/roll/ne/%7Bfirst/secondtolast/subgrid%7Dif/dup//searchDaysCess/get_parameter/add/4/-3/roll/replaceGRID/exch/replaceNaN//pointwidth/1/def//long_name/(Cessation%20Date%20since%20)//earlyCess/get_parameter/append/def/T/dup/yearlyedgesgrid/first/secondtolast/subgrid/replaceGRID/T/(months%20since%201960-01-01)/streamgridunitconvert/T//pointwidth/12/def/6/shiftGRID/info.json?"
    onset_json = json.loads(urllib.request.urlopen(onset_url).read().decode())
    cess_json = json.loads(urllib.request.urlopen(cess_url).read().decode())
    years = [ i['T'] for i in onset_json['iridl:values'] ]
    onset_data = [ i['min'][:-4] for i in onset_json['iridl:values'] ]
    cess_data = [ i['min'][:-4] for i in cess_json['iridl:values'] ]
    return [ [ years[i], onset_data[i], cess_data[i] ] for i in range(len(years)) ]
