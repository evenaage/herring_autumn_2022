import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import torch
from netCDF4 import Dataset



def print_on_map(y,  catch_data = None,  figsize = (10,10), show_depth=None):
    '''
    Prints a prediction on a map of the Norwegian sea in the form of a probability map. Optionally includes catch data, which are printed on top.
    Functions as a human interpretable model validation tool.
    
        Parameters 
        ----------
            y : torch.tensor 
                A 620x941 tensor with predictions for herring distribution probabilities, with grid cells corresponding to the sinmod simulation mapping. Elements must be in [0, inf]

            catch_data :pd.DataFrame, optional
                Pandas DataFrame with required columns: 'latitude', 'longitude', 'rundvekt'. Plots on top of heatmap.

            depth : bool, optional
                Bool to plot depth contours.

            figsize: tuple (int, int), optional
                Figure size in inches, default is (10,10)
        Returns
        -------
            None
    '''

    #load necessary masks to filter out land and sea, in addition to depth and translation to latlon from grid cells
    path = os.path.dirname(__file__)
    mask = np.load(os.path.join(path ,'sinmod_land_sea_mask'), allow_pickle=True)
    lats = np.load(os.path.join(path ,'lats'), allow_pickle=True)
    lons = np.load(os.path.join(path ,'lons'), allow_pickle=True)
    depth = np.load(os.path.join(path ,'depth'), allow_pickle=True)

    #figure stuff, also load world map with correct latitudes and longitudes
    fig = plt.figure()
    fig.set_size_inches(figsize[0] , figsize[1])
    map = Basemap(projection='merc', llcrnrlon=-15.,llcrnrlat=51.5,urcrnrlon=35.,urcrnrlat=75.,resolution='i' )
    
    #prediction y is either a tensor or a numpy array, and needs to be a numpy array
    if type(y) == torch.tensor:
        pred_numpy= y.detach().numpy()[0,:,:]
    else:
        pred_numpy = y[0]

    pred_normalized = pred_numpy/pred_numpy.max()

    #
    probabilities = map.contourf(lons, lats,np.ma.masked_array(pred_normalized, mask= mask),100, latlon=True, cmap =plt.cm.RdYlBu_r)
    cb = map.colorbar(probabilities,"bottom", size="5%", pad="2%", label="Probability 0-1")

    #plot catch data, if possible.
    try:  
        map.scatter( catch_data['latitude'], catch_data['longitude'],latlon=True, marker='^')
    except:
        pass
    if show_depth: map.contour(lons, lats, depth,latlon=True)

    #draw world map
    map.etopo()

def get_relevant_catch_data_csv(path_to_dataset, year = None, month = None, day = None):
    '''
    A function to filter out the relevant data from a .csv file, based on date or date range. 

    Parameters
    ----------
    path_to_dataset : str
        Path to the dataset that is to be filtered. Has to be path to a .csv file, not a folder.
    year(s) : int, tuple, list, optional
        Year of data to acquire. If not provided, funcion will ignore year.
    month(s) : int, tuple, list, optional
        Month of data to acquire. If not provided, function will ignore month.
    day(s) : int, tuple, list, optional
        Day of data to acquire. If not provided, function will ignore day. 

    Returns
    -------
        data : pd.DataFrame
            The acquired data. Columns in dataframe include: 'latitude', 'longitude', 'rundvekt' and 'start_date' (str) and 'end_date', in addition to 'year' (int), month (int), day(int) and hour(s), if provided. Other columns depend on what type of data is read.

    Raises
    ------
    IllegalArgumentException
        Year, month, day, or path to dataset wrong, so function cannot return any data.

    '''

def get_relevant_data_nc(path_to_dataset, year = None, month = None, day = None, hour = None, DEPTH_LAYERS=15):
    '''
    A function to filter out the relevant data from a .nc file, in order to match them to catch data.

    Parameters
    ----------
    path_to_dataset : str
        Path to the dataset that is to be filtered. Has to be path to a folder of .nc files on the format: 'samples_year_month_date.nc.
    year(s) : int, tuple, list, optional
        Year(s) of data to acquire. If not provided, funcion will ignore year.
    month(s) : int, tuple, list, optional
        Month(s) of data to acquire. If not provided, function will ignore month.
    day(s) : int, tuple, list, optional
        Day(s) of data to acquire. If not provided, function will ignore day. 
    hour(s): int, tuple, list, optional
        Hour(s) of data to acquire. If not provided, function will ignore hour. 
    Returns
    -------
       data : np.array DEPTH_LAYERSx5
 
    Raises
    ------
    IllegalArgumentException
        Year, month, day, or path to dataset wrong, so function cannot return any data.

    '''
    pass
    nc = Dataset(path_to_dataset)
    lats = nc.variables['gridLats'][:]
    lons = nc.variables['gridLons'][:]
    time = nc.variables['time'][:]
    temp = nc.variables['temperature'][:]
    u_current = nc.variables['u_east'][:]
    v_current = nc.variables['v_north'][:]
    u_wind = nc.variables['w_east'][:]
    v_wind = nc.variables['w_north'][:]
    w_velocity =nc.variables['w_velocity'][:]


if __name__ == '__main__': 
    pred = np.load('test_prediction', allow_pickle=True)
    print_on_map(pred)
    plt.show()
    #get_relevant_data_csv()