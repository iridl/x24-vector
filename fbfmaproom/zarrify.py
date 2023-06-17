import xarray as xr

SOURCE_ROOT = Path('/')
DEST_ROOT = Path('/data/aaron/fbf-test')
datasets = [
    [
        'data/aaron/NigerFBF2023/prcp/jas'
        'niger/prcp/jas-v4'
    ],
]
    
hindcast_file = SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/Hindcasts/June/NextGen_PRCP_Jul-Sep_iniJun.tsv'
obs_file = SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/obs_PRCP_Jul-Sep.tsv'
mu_file_pattern = SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/Forecast_mu/June/NextGen_PRCPPRCP_CCAFCST_mu_Jul-Sep_Jun*.tsv'
var_file_pattern = SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/Forecast_var/June/NextGen_PRCPPRCP_CCAFCST_var_Jul-Sep_Jun*.tsv'

hindcasts = xr.open_dataset(hindcast_file)
print(hindcasts)
