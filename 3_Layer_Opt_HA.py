#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load in neccessary functions

import xarray as xr
import dask
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import binned_statistic
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime


# In[3]:


#local radar_equivalent_function
import res_function as iso

import warnings
warnings.filterwarnings('ignore', category=ResourceWarning)


# In[4]:


# Load in SMRT functions 

from smrt.core.globalconstants import DENSITY_OF_ICE
from smrt import sensor_list, make_model, make_snowpack, make_interface
from smrt.emmodel import iba
from smrt.substrate.reflector_backscatter import make_reflector
from smrt.utils import dB


# In[5]:


# Define debye relationship

def debye_eqn(ssa, density):

    return 4 * (1 - density / DENSITY_OF_ICE) / (ssa * DENSITY_OF_ICE)

# In[6]

# Define the path where the files are located
file_path = r'Havikpak_Arctic'

# Loop through all files in the directory and subdirectories
for root, dirs, files in os.walk(file_path):
    for file in files:
        # Check if the file starts with 'Converted'
        if file.startswith('OptDiam'):
            # Get the full file path
            full_file_path = os.path.join(root, file)
            
            # Remove the file
            try:
                os.remove(full_file_path)
                print(f"Removed file: {full_file_path}")
            except Exception as e:
                print(f"Error removing file {full_file_path}: {e}")
                
# In[7]

# Define the path where the files are located
file_path = r'Havikpak_Arctic'

# Loop through all files in the directory and subdirectories
for root, dirs, files in os.walk(file_path):
    for file in files:
        # Check if the file starts with 'Converted'
        if file.startswith('Converted'):
            # Get the full file path
            full_file_path = os.path.join(root, file)
            
            # Remove the file
            try:
                os.remove(full_file_path)
                print(f"Removed file: {full_file_path}")
            except Exception as e:
                print(f"Error removing file {full_file_path}: {e}")


# In[21]:


def import_crocus(file_path, str_year_begin):
 
    mod = xr.open_dataset(file_path)

    # Convert the 'time' variable to datetime format
    mod['time'] = xr.cftime_range(
        start="2015-09-02 07:00:00", 
        periods=mod.dims['time'], 
        freq="H", 
        calendar="proleptic_gregorian"
    )
    
    mod['time'] = pd.to_datetime(mod.indexes['time'].to_datetimeindex())
    

    # Convert selected variables to DataFrame
    df = mod[['SNODEN_ML', 'SNOMA_ML', 'TSNOW_ML', 'SNODOPT_ML', 'SNODP','time']].to_dataframe().dropna()
    
    # Define the start date 
    str_begin = str_year_begin + '-09-02 07:00:00'
    sep01 = pd.to_datetime(str_begin)
    
    print(sep01) 
    
    # Filter DataFrame to start from September 1 
    df = df[df.index.get_level_values('time') >= sep01]

    # Calculate 'thickness' as SNOMA_ML / SNODEN_ML

    df['thickness'] = df['SNOMA_ML'] / df['SNODEN_ML']
    
    df['ssa'] = df['SNODOPT_ML'].where(df['SNODOPT_ML'] > 0, 0).apply(lambda x: 6 / (x * 917) if x > 0 else 0)

    #df['thickness'] = df[['SNODEN_ML', 'SNOMA_ML']].apply(lambda x: x[1] / x[0], axis=1)

    # Calculate SSA: 6 / (SNODOPT_ML * 917) if SNODOPT_ML > 0
    #df['ssa'] = df['SNODOPT_ML'].apply(lambda x: 6 / (x * 917) if x > 0 else 0)

    # Filter out rows with low snow depth (SNODP) and small snow layers (thickness)
    df = df[(df.SNODP > 0.10) & (df.thickness > 0.005)]

    # Get unique dates
    dates = df.index.get_level_values('time').unique()
    
    # Add 'height' column to the DataFrame
    df['height'] = np.nan

    # Calculate the 'height' for each timestamp by applying cumulative sum in reverse for each date
    #for date in dates:
    
    #    df_temp = df.xs(date, level='time', drop_level=False)
        
  
    #    df.loc[df_temp.index, 'height'] = np.cumsum(df_temp['thickness'].values[::-1])[::-1]

    df['height'] = df.groupby('time')['thickness'].cumsum()

    return df, dates


# In[22]:


# Define the main directory where subdirectories and files are located
main_directory = r'Havikpak_Arctic'

# Define the function arguments
str_year_begin = '2015'

# Loop through all subdirectories and files
for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.endswith('.nc'):  
            file_path = os.path.join(root, file)
            
            # Use context manager to open the NetCDF file and ensure it's closed after use
            with xr.open_dataset(file_path) as xr_tvc:
                
                # Call the function for each file
                df, dates = import_crocus(file_path, str_year_begin)
                
                # Convert the DataFrame to an xarray.Dataset
                df_reset = df.reset_index()
                
                # Create a Dataset
                ds = xr.Dataset(
                    {
                        "thickness": ("time", df_reset["thickness"].values),
                        "ssa": ("time", df_reset["ssa"].values),
                        "height": ("time", df_reset["height"].values),
                        "SNODEN_ML": ("time", df_reset["SNODEN_ML"].values),
                        "SNOMA_ML": ("time", df_reset["SNOMA_ML"].values),
                        "TSNOW_ML": ("time", df_reset["TSNOW_ML"].values),
                        "SNODOPT_ML": ("time", df_reset["SNODOPT_ML"].values),
                        "SNODP": ("time", df_reset["SNODP"].values),
                    },
                    coords={
                        "time": df_reset["time"].values,
                    },
                )
                
                # Create a unique output file path
                output_file_name = 'OptDiam_{}.nc'.format(os.path.splitext(file)[0])
                output_file_path = os.path.join(root, output_file_name)
                
                # Save the dataset to a NetCDF file
                ds.to_netcdf(output_file_path)

            print(f"Data saved to {output_file_path}")


# In[23]:


def three_layer_k(snow_df, method = 'thick-ke-density', freq = 13.25e9):
    """
    Kmeans 3 cluster method
    method param :str that need indicate average method
    freq: float for frequency of sensor, defaut is TSMM upper Ku
    """
    X = pd.DataFrame({ 'ke' : compute_ke(snow_df, freq =freq),  'height' : snow_df.height})
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
    snow_df['label'] = kmeans.labels_
    
    df = snow_df.groupby('label', sort = False).apply(lambda x: avg_snow_sum_thick(x, method = method, freq =freq))
    return df


# In[24]:


def compute_ke(snow_df, freq = 13.25e9):
    """
    add ke to the snow dataframe
    freq : frequency at which ke is calculated
    """
    if isinstance(snow_df.thickness, np.floating):
        thickness = [snow_df.thickness]
    else:
        thickness = snow_df.thickness

    sp = make_snowpack(thickness=thickness, 
                        microstructure_model='exponential',
                        density= snow_df.SNODEN_ML,
                        temperature= snow_df.TSNOW_ML,
                        corr_length = debye_eqn(np.array(snow_df.ssa), np.array(snow_df.SNODEN_ML)))
    #create sensor
    sensor  = sensor_list.active(freq, 35)
    
    #get ks from IBA class
    ks = np.array([iba.IBA(sensor, layer, dense_snow_correction='auto').ks for layer in sp.layers])
    ka = np.array([iba.IBA(sensor, layer, dense_snow_correction='auto').ka for layer in sp.layers])
    ke = ks + ka
    return ke


# In[25]:


def avg_snow_sum_thick(snow_df, method = 'thick-ke-density', freq = 13.25e9):
    """
    Averaging method
    method param :str that need indicate average method
    """
    thick = snow_df.thickness.sum()
    if method == 'thick':
        snow_mean = snow_df.apply(lambda x: np.average(x, weights = snow_df.thickness.values), axis =0)
        snow_mean['thickness'] = thick
        return snow_mean
    if method == 'thick-ke':
        snow_df['ke'] = compute_ke(snow_df, freq = freq)
        snow_mean = snow_df.apply(lambda x: np.average(x, weights = snow_df.thickness.values * snow_df.ke.values), axis =0)
        snow_mean['thickness'] = thick
        return snow_mean
    if method == 'thick-ke-density':
        snow_df['ke'] = compute_ke(snow_df, freq = freq)
        df_copy = snow_df.copy()
        density_temp = np.average(df_copy.SNODEN_ML, weights = snow_df.thickness.values )
        snow_mean = snow_df.apply(lambda x: np.average(x, weights =  snow_df.thickness.values*snow_df.ke.values, axis =0))
        snow_mean['thickness'] = thick
        snow_mean['SNODEN_ML'] = density_temp
        return snow_mean
    else:
        print('provide a valid method')
        return np.nan


# In[29]:


main_directory = r'Havikpak_Arctic'
time_value = pd.to_datetime('2022-03-18T12:00:00')

data_storage = {}

for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.startswith('OptDiam') and file.endswith('.nc'):
            file_path = os.path.join(root, file)

            # Use context manager to handle file
            with xr.open_dataset(file_path) as xr_tvc:
                specific_data = xr_tvc.sel(time=time_value)

                # Check and modify 'SNODOPT_ML' safely
                if 'SNODOPT_ML' in specific_data.variables:
                    specific_data = specific_data.copy()  # Work on a copy
                    specific_data['SNODOPT_ML'] = specific_data['SNODOPT_ML'].where(
                        specific_data['SNODOPT_ML'] <= 0.00075, 0.00075
                    )

                # Convert to DataFrame after reducing dimensions
                df_tvc = specific_data.to_dataframe().dropna()
                data_storage[file_path] = df_tvc

                print(f"Data for {file_path}:\n{df_tvc}")

example_file = list(data_storage.keys())[0]
print(f"\nAccessed data for {example_file}:\n{data_storage[example_file]}")


# In[30]:


root_dir = r"Havikpak_Arctic"

output_directory = r"Rescaled/OptDiam/Havikpak_Arctic"
os.makedirs(output_directory, exist_ok=True) 

# Loop through each DataFrame in data_storage and apply the 3-layer scaling method
for file_path, df_tvc in data_storage.items():
    
    result_df = three_layer_k(df_tvc, method='thick-ke-density', freq=13.25e9)
    
    # Extract the relevant path components to make a unique filename
    original_file_name = os.path.basename(file_path)  
    base_file_name = os.path.splitext(original_file_name)[0]  
    
    # Get the full folder path 
    parent_folder = os.path.basename(os.path.dirname(file_path)) 
    
    # Create the output file name 
    output_file_name = 'OptDiam_' + parent_folder + '.nc'
    
    # Full path for saving the file
    output_file_path = os.path.join(output_directory, output_file_name)
    
    # Save the result DataFrame to a NetCDF file 
    result_ds = result_df.to_xarray()  
    result_ds.to_netcdf(output_file_path) 
    
    print(f"Results saved to {output_file_path}")

print("\nAll files processed and saved.")


# In[ ]:




