import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
from plotly.io import write_html

def unit_conversion(ds, variable):
     #if precipitation variable, change from kg to mm per day
    if variable == 'pr':
        ds[variable] *= 86400
        ds[variable].attrs['units'] = 'mm/day' #rename unit to converted
    elif variable in ['tas', 'tasmin', 'tasmax']:
        ds[variable] -= 273.15 
        ds[variable].attrs['units'] = 'Celsius'
    
    return ds 

def compute_monthly_avg(ds, variable):
    # Convert daily data to monthly
    monthly_ds = ds.resample(T="1M").mean()

    # spatial conversion
    monthly_avg = monthly_ds.mean(dim=["X", "Y"])

    return monthly_avg


def compute_annual_monthly_avg(monthly_avg, variable):
    #compute the avg (variable) for each month across the specified years

    #Compute the average for each month over time
    annual_monthly_avg = monthly_avg.groupby("T.month").mean(dim="T")
    return annual_monthly_avg

def compute_seasonal_avg(monthly_avg):
    rolling_avg = monthly_avg.rolling(T=3, center=True).mean()
    return rolling_avg
    
def compute_annual_seasonal_avg(rolling_avg, variable):
    
    # Compute the average variable for each rolling season across all years
    rolling_seasonal_avg = rolling_avg.groupby("T.month").mean(dim="T")
    return rolling_seasonal_avg


def create_seasonal_dataframe(rolling_seasonal_avg):
    # Create a DataFrame for seasonal averages
    seasonal_avgs_df = rolling_seasonal_avg.to_dataframe().reset_index()

    # Assign the season column with correct length
    seasonal_avgs_df['season'] = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                                   'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

    return seasonal_avgs_df


def write_xarray_to_csv(ds, scenario, model, variable, output_dir):
    df = ds.to_dataframe().reset_index()
    file_path = f'{output_dir}/{scenario}_{model}_{variable}_annual_monthly_avg.csv'
    df.to_csv(file_path, index=False)

def write_dataframe_to_csv(df, scenario, model, variable, output_dir):
    file_path = f'{output_dir}/{scenario}_{model}_{variable}_seasonal_avg.csv'
    df.to_csv(file_path, index=False)

'''
def map_averages(ds, variable):
    
    #different than compute_annual_monthly_avg because keeps the spatial component when averaging
    #this will be inputted into the plotting function
    

    # Convert daily data to monthly
    monthly_ds = ds.resample(T="1M").mean()

    # Compute the average for each month across all years
    monthly_avg = monthly_ds.groupby("T.month").mean(dim="T")

    return monthly_avg


def plot_monthly(ds, variable, scenario, model, output_dir):
    #convert units if needed
    units = ds[variable].attrs.get('units', 'unknown')
    
    #print all 12 months (12 different maps)
    for month in range(1, 13):
        monthly_data = ds.sel(month=month)
        fig = px.imshow(
            monthly_data[variable],
            labels={'color': f'{variable} ({units})'},
            title=f'Monthly Average ({variable})  - Month {month}',
            origin='lower'
        )
        #fig.show() print automatically
        output_file = f"{output_dir}/{variable}_{scenario}_{model}_monthly_avg_map_{month}.html"
        fig.write_html(file=output_file)
        print("Saved to output directory")
  
'''
def main(scenario, model, variable, start_year=None, end_year=None, output_dir='/home/sz3116/outputs'):
    #main to run functions

    # Open zarr file, read data 
    ds = xr.open_zarr(f'//Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/{scenario}/{model}/zarr/{variable}')

    #select time section
    ds = ds.sel(T=slice(f"{start_year}", f"{end_year}"))
    
    #apply unit conversion if necessary
    ds = unit_conversion(ds, variable)
    
    # Compute monthly averages
    monthly_ds = compute_monthly_avg(ds, variable)
    
    # Compute annual monthly averages
    annual_monthly_avg = compute_annual_monthly_avg(monthly_ds, variable, )
    
    rolling_avg = compute_seasonal_avg(monthly_ds)
    annual_seasonal_avg = compute_annual_seasonal_avg(rolling_avg, variable)
    #call seasonal function
    #rolling_seasonal_avg = compute_annual_seasonal_avg(ds, variable)

    seasonal_df = create_seasonal_dataframe(annual_seasonal_avg)
    
    
    # Write the data to a CSV file
    write_xarray_to_csv(annual_monthly_avg, scenario, model, variable, output_dir)
    write_dataframe_to_csv(seasonal_df, scenario, model, f'{variable}_seasonal_', output_dir)
    
    #Get monthly avgs for mapping
    #plotting_monthly = map_averages(ds, variable)
    #Generate map
    #plot_monthly(plotting_monthly, variable, scenario , model, output_dir)
    
    
    #also print in terminal
    print(annual_monthly_avg)
    print(annual_seasonal_avg)
    

# testing
if __name__ == "__main__":
    scenario = "historical"     #input options: ssp126,  ssp370, ssp585, historical
    model = "GFDL-ESM4"     #GFDL-ESM4,  IPSL-CM6A-LR,  MPI-ESM1-2-HR,  MRI-ESM2-0,  UKESM1-0-LL
    variable = "pr"     #tas, tasmin, tasmax, pr, rlds
    start_year = 1950
    end_year = 2014
    output_dir = '/home/sz3116/outputs'

    main(scenario, model, variable, start_year, end_year, output_dir)

print("done")
