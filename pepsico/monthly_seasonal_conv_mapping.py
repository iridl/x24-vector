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
    #compute seasonal averages
    rolling_avg = monthly_avg.rolling(T=3, center=True).mean()
    return rolling_avg
    
def compute_annual_seasonal_avg(rolling_avg, variable):
    
    # Compute the average variable for each rolling season across all years
    rolling_seasonal_avg = rolling_avg.groupby("T.month").mean(dim="T")
    return rolling_seasonal_avg

def compute_seasonal_anomalies(historical_ds, future_ds, variable):
    # Compute monthly averages
    historical_monthly_avg = compute_monthly_avg(historical_ds, variable)
    future_monthly_avg = compute_monthly_avg(future_ds, variable)
    
    #compute seasonal avg
    historical_seasonal_avg = compute_seasonal_avg(historical_monthly_avg)
    future_seasonal_avg = compute_seasonal_avg(future_monthly_avg)
    
    #compute seasonal avg over T(time)
    historical_rolling = compute_annual_seasonal_avg(historical_seasonal_avg, variable)
    future_rolling = compute_annual_seasonal_avg(future_seasonal_avg, variable)

    historical_computed = historical_rolling.compute()
    future_computed = future_rolling.compute()
    print("Historical")
    print(historical_computed)
    print("Future")
    print(future_computed)
    anomalies = future_rolling - historical_rolling

    return anomalies


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

def map_averages(ds, variable):
    
    #different than compute_annual_monthly_avg because keeps the spatial component when averaging
    #this will be inputted into the plotting function
    
    # Convert daily data to monthly
    monthly_ds = ds.resample(T="1M").mean()

    # Compute the average for each month across all years
    monthly_avg = monthly_ds.groupby("T.month").mean(dim="T")

    return monthly_avg

def calc_seasonal_plot(historical_ds, future_ds, variable):
    # Compute monthly averages
    historical_monthly_avg = historical_ds.resample(T="1M").mean().groupby("T.month").mean(dim="T")
    future_monthly_avg = future_ds.resample(T="1M").mean().groupby("T.month").mean(dim="T")
    
    # Compute anomalies
    monthly_anomalies = future_monthly_avg - historical_monthly_avg
    
    # Compute seasonal averages of anomalies
    seasonal_anomalies = monthly_anomalies.rolling(month=3, center=True).mean()
    
    return seasonal_anomalies 

'''
def plot_seasonal_anomalies(map_anomalies, variable, scenario, model, output_dir, season):

    units = map_anomalies[variable].attrs.get('units', 'unknown')
    # Determine season indices based on the selected season
    season_indices = {
        'DJF': [11, 0, 1],  # Dec, Jan, Feb
        'JFM': [0, 1, 2],   # Jan, Feb, Mar
        'FMA': [1, 2, 3],   # Feb, Mar, Apr
        'MAM': [2, 3, 4],   # Mar, Apr, May
        'AMJ': [3, 4, 5],   # Apr, May, Jun
        'MJJ': [4, 5, 6],   # May, Jun, Jul
        'JJA': [5, 6, 7],   # Jun, Jul, Aug
        'JAS': [6, 7, 8],   # Jul, Aug, Sep
        'ASO': [7, 8, 9],   # Aug, Sep, Oct
        'SON': [8, 9, 10],  # Sep, Oct, Nov
        'OND': [9, 10, 11], # Oct, Nov, Dec
        'NDJ': [10, 11, 0]  # Nov, Dec, Jan
    }
    # Select data for the specific season
    seasonal_data = map_anomalies.sel(T=map_anomalies['T.month'].isin(season_indices[season]))
    seasonal_avg = seasonal_data.mean(dim='T')
    
    # Plot the seasonal average data
    fig = px.imshow(
        seasonal_anomalies[variable],
        labels={'color': f'{variable} anomaly ({units})'},
        title=f'Seasonal Anomalies ({variable}) - Season {season}',
        origin='lower',
        color_continuous_scale='RdBu_r',  # Red-Blue diverging colorscale
        color_continuous_midpoint=0  # Center the colorscale at 0
    ) 
    
    # Save the plot to an HTML file
    output_file = f"{output_dir}/{variable}_{scenario}_{model}_seasonal_anomalies_map_season_{season}.html"
    fig.write_html(file=output_file)
    fig.show()
    print(f"Saved anomalies map to {output_file}")
'''
    

def plot_monthly(ds, variable, scenario, model, output_dir, month):
    #convert units if needed
    units = ds[variable].attrs.get('units', 'unknown')
    
    monthly_data = ds.sel(month=month)
    
    map_zmin = None
    map_zmax = None

    if variable == 'pr':
        map_zmin = 0
        map_zmax = 15

    fig = px.imshow(
        monthly_data[variable],
        labels={'color': f'{variable} ({units})'},
        title=f'Monthly Average ({variable})  - Month {month}',
        origin='lower',
        zmin=map_zmin,
        zmax=map_zmax
    )

    fig.show() 
    output_file = f"{output_dir}/{variable}_{scenario}_{model}_monthly_avg_map_{month}.html"
    fig.write_html(file=output_file)
    print("Printed")



def plot_seasonal(ds, variable, scenario, model, output_dir, season):
    units = ds[variable].attrs.get('units', 'unknown')
    # Determine season indices based on the selected season
    season_indices = {
        'DJF': [11, 0, 1],  # Dec, Jan, Feb
        'JFM': [0, 1, 2],   # Jan, Feb, Mar
        'FMA': [1, 2, 3],   # Feb, Mar, Apr
        'MAM': [2, 3, 4],   # Mar, Apr, May
        'AMJ': [3, 4, 5],   # Apr, May, Jun
        'MJJ': [4, 5, 6],   # May, Jun, Jul
        'JJA': [5, 6, 7],   # Jun, Jul, Aug
        'JAS': [6, 7, 8],   # Jul, Aug, Sep
        'ASO': [7, 8, 9],   # Aug, Sep, Oct
        'SON': [8, 9, 10],  # Sep, Oct, Nov
        'OND': [9, 10, 11], # Oct, Nov, Dec
        'NDJ': [10, 11, 0]  # Nov, Dec, Jan
    }
    # Select data for the specific season
    seasonal_data = ds.sel(T=ds['T.month'].isin(season_indices[season]))
    seasonal_avg = seasonal_data.mean(dim='T')
    
    map_zmin = None
    map_zmax = None

    if variable == 'pr':
        map_zmin = 0
        map_zmax = 15

    fig = px.imshow(
        seasonal_avg[variable],
        labels={'color': f'{variable} ({units})'},
        title=f'Seasonal Average ({variable})  - Season {season}',
        origin='lower',
        zmin=map_zmin,
        zmax=map_zmax
    )
    # Save the plot to an HTML file
    output_file = f"{output_dir}/{variable}_{scenario}_{model}_seasonal_avg_map_season_{season}.html"
    fig.write_html(file=output_file)
    fig.show()
    print(f"Saved to {output_file}")



def main(scenario, model, variable, start_year=None, end_year=None, output_dir='/home/sz3116/outputs'):
    #main to run functions

    # Open zarr file, read data 
    ds = xr.open_zarr(f'/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/{scenario}/{model}/zarr/{variable}')

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
    

    seasonal_df = create_seasonal_dataframe(annual_seasonal_avg)
    
    
    # Write the data to a CSV file
    write_xarray_to_csv(annual_monthly_avg, scenario, model, variable, output_dir)
    write_dataframe_to_csv(seasonal_df, scenario, model, f'{variable}_seasonal_', output_dir)
   
    #also print in terminal
    print(annual_monthly_avg)
    print(annual_seasonal_avg)

    #Get monthly avgs for mapping
    plotting_monthly = map_averages(ds, variable)
    #Generate map (choose month) (e.g., 3=March)
    plot_monthly(plotting_monthly, variable, scenario , model, output_dir, month=3)

    # Ploting seasonal averages for a specific season (e.g., JFM)
    selected_season = 'FMA'  
    plot_seasonal(ds, variable, scenario, model, output_dir, selected_season)
    
    '''
    # Compute seasonal anomalies if scenario is not historical
    if scenario != "historical":
        historical_ds = xr.open_zarr(f'//Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/{model}/zarr/{variable}')
        historical_ds = historical_ds.sel(T=slice("1981", "2014"))
        historical_ds = unit_conversion(historical_ds, variable)
        anomalies = compute_seasonal_anomalies(historical_ds, ds, variable)
        
        # Print anomalies
        print("Seasonal Anomalies:")
        print(anomalies)
        
        # Save anomalies to CSV
        anomalies_df = anomalies.to_dataframe().reset_index()
        anomalies_file_path = f'{output_dir}/{variable}_{scenario}_{model}_seasonal_anomalies.csv'
        anomalies_df.to_csv(anomalies_file_path, index=False)
        print(f"Anomalies saved to {anomalies_file_path}")

        map_anomalies = calc_seasonal_plot(historical_ds, ds, variable)
        plot_seasonal_anomalies(map_anomalies, variable, scenario, model, output_dir, selected_season)
        print("Mapped anomalies")
    '''

# testing
if __name__ == "__main__":
    scenario = "ssp585"     #input options: ssp126,  ssp370, ssp585, historical
    model = "GFDL-ESM4"     #GFDL-ESM4,  IPSL-CM6A-LR,  MPI-ESM1-2-HR,  MRI-ESM2-0,  UKESM1-0-LL
    variable = "tasmin"     #tas, tasmin, tasmax, pr, rlds
    start_year = 2030
    end_year = 2034
    output_dir = '/home/sz3116/outputs'

    main(scenario, model, variable, start_year, end_year, output_dir)

print("done")
