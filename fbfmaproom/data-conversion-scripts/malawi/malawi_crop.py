from cftime import Datetime360Day as DT360
import pandas as pd

def translate_fnid(fnid):
    return keys[fnid]

def translate_year(year):
    return DT360(year, 1, 16)


keys = {
    # DesignEngine=> select '"' || adm2_name || '": "' || (adm0_code, adm1_code, adm2_code) || '",' from iridb.g2015_2014_2 where adm0_name = 'Malawi' order by 1;
 "Area under National Administration": "(152,65268,65269)",
 "Balaka": "(152,1890,42174)",
 "Blantyre": "(152,1890,19321)",
 "Chikwawa": "(152,1890,19322)",
 "Chiradzulu": "(152,1890,19323)",
 "Chitipa": "(152,1889,19316)",
 "Dedza": "(152,1888,19307)",
 "Dowa": "(152,1888,19308)",
 "Karonga": "(152,1889,19317)",
 "Kasungu": "(152,1888,19309)",
 "Likoma": "(152,1889,42172)",
 "Lilongwe": "(152,1888,19310)",
 "Machinga": "(152,1890,42175)",
 "Mangochi": "(152,1890,19325)",
 "Mchinji": "(152,1888,19311)",
 "Mulanje": "(152,1890,42176)",
 "Mwanza": "(152,1890,65271)",
 "Mzimba": "(152,1889,19318)",
 "Neno": "(152,1890,65270)",
 "Nkhata Bay": "(152,1889,42173)",
 "Nkhotakota": "(152,1888,19312)",
 "Nsanje": "(152,1890,19328)",
 "Ntcheu": "(152,1888,19313)",
 "Ntchisi": "(152,1888,19314)",
 "Phalombe": "(152,1890,42177)",
 "Rumphi": "(152,1889,19320)",
 "Salima": "(152,1888,19315)",
 "Thyolo": "(152,1890,19329)",
 "Zomba": "(152,1890,19330)",
    
}


ds = (
    pd.read_csv('malawi_hindcast_output.csv')
    [lambda x: x['month'] == 2]
    .assign(
        time=lambda x: x['year'].apply(translate_year),
        geom_key=lambda x: x['admin2'].apply(translate_fnid),
    )
    .set_index(['geom_key', 'time'])
    [['value']]
    .to_xarray()
)
print(ds)
ds.to_zarr('malawi-crop-forecast-feb.zarr')
