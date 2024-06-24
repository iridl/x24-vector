import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px

#for hurs variable

#open zarr file 
ds = xr.open_zarr('/data/remic/mydatafiles/zarr/test/hurs')

print(ds.head())


#convert daily data to monthly
monthly_ds = ds.resample(T="1M").mean()

#spatial avg for each month
monthly_avg_hurs = monthly_ds['hurs'].mean(dim=["X", "Y"])
#print(monthly_ds.head())

# Select a specific month (e.g., January 1951)
#specific_month = '2014-01'
# Extract the data for the specific month + squeeze the extra dimension
#hurs_specific_month = monthly_ds['hurs'].sel(T=specific_month).compute()


# Convert to Pandas DataFrame for use with Plotly
df = monthly_ds.to_dataframe().reset_index()

'''
# Change this to your desired file path
df.to_csv('/home/sz3116/python-maprooms/pepsico/resources/joe.csv', index=False)



# Create the spatial plot
fig = px.imshow(hurs_specific_month.values, 
                labels={'color': 'Relative Humidity'},
                x=monthly_ds['X'].values,
                y=monthly_ds['Y'].values,
                title=f'Relative Humidity for {specific_month}')
fig.update_layout(
    xaxis_title='Longitude',
    yaxis_title='Latitude'
)
fig.show()
'''
'''
#print the entire dataframe
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Monthly averages for hurs:")
    print(df)

# Save to CSV (optional)
'''
print("testing")
print(df.head(800))
df.to_csv('/home/sz3116/python-maprooms/pepsico/resources/monthly_hurs.csv', index=False)
print("done")
#print("monthly")
#print(df)