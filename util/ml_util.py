import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from numpy import sin, cos,sqrt, arctan, arctan2
import os
import torch
from netCDF4 import Dataset
import pandas as pd
from sklearn.linear_model import LinearRegression



def ll2xy(lat, lon):
    '''
    Function to translate from latitude and longitude into x and y coordinates for sinmod grid cells.
    Translated from matlab.

    Parameters
    ----------
        lat : float
            latitude to translate into grid cell x
        lon : float
            longitude to translate into grid cell y
    
    Returns:
        x : int
            grid cell x value
        y : int 
            grid cell y value
    '''
    EARTHRAD = 6371
    GridData = [826, 638.5000, 4, 58, 60, 1]
    RAD = np.pi/180
    AN = EARTHRAD*(1+np.sin(abs(GridData[4]))*RAD)/GridData[2]
    VXR = GridData[3]*RAD
    PLR=lon*RAD
    PBR=lat*RAD
    R = AN*np.cos(PBR)/(1+np.sin(PBR))
    x=R*(np.sin(PLR)*np.cos(VXR)-np.sin(VXR)*np.cos(PLR))+GridData[0]
    y=-R*(np.cos(PLR)*np.cos(VXR)+np.sin(PLR)*np.sin(VXR))+GridData[1]
    return int(x), int(y)

def xy2ll(x, y):
    '''
    Function to translate from x and y to longitude and latitude.
    Translated from matlab.

    Parameters
    ----------
        x : int
            Grid position x
        y : int
            Grid position y

    Returns
    -------
        lat : float
            latitude of x coordinate
        lon : float
            longitude of y coordinate
        
    '''
    GridData = [826, 638.5000, 4, 58, 60, 1]
    R = 6370
    OMRE=180/3.14159265
    AN= R*(1+sin(abs(GridData[4])/OMRE))/GridData[2]
    RP= sqrt((x-GridData[0])**2 + (y-GridData[1])**2)
    B = 90-2*OMRE*arctan(RP/AN)
    L = GridData[3]+OMRE*arctan2(x-GridData[0],GridData[1]-y)
    if L > 180:
        L -= 360
    if L < - 180:
        L += 360
    return B, L


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
        pred_numpy = y

    pred_normalized = pred_numpy/pred_numpy.max()

    #
    probabilities = map.contourf(lons, lats,np.ma.masked_array(pred_normalized, mask= mask),100, latlon=True, cmap =plt.cm.RdYlBu_r)
    cb = map.colorbar(probabilities,"bottom", size="5%", pad="2%", label="Probability 0-1")

    #plot catch data, if possible.
    try:  
        map.scatter( catch_data['longitude'], catch_data['latitude'],latlon=True, marker='^')
    except:
        print("scatter didnt work")
        pass
    if show_depth: map.contour(lons, lats, depth,latlon=True)

    #draw world map
    map.etopo()

def get_relevant_catch_data_csv(path_to_dataset, year, month, day = None, hour = None):
    '''
    A function to filter out the relevant data from a .csv file, based on date or date range. 

    Parameters
    ----------
    path_to_dataset : str
        Path to the dataset that is to be filtered. Has to be path to a .csv file, not a folder.
    year : int
        Year of data to acquire. 
    month : int
        Month of data to acquire. If not provided, function will ignore month.
    day(s) : int, tuple, list
        Day of data to acquire. If not provided, function will ignore day. 
    hour(s): int(s) in [0,23], or a list of hours 
        CURRENTLY NOT IMPLEMENTED
        Hour(s) of data to acquire. If not provided, function will aquire all data from day. 
    Returns
    -------
        data : pd.DataFrame
            The acquired data. Columns in dataframe include: 'latitude', 'longitude', 'rundvekt' and 'start_date' (str) and 'end_date', in addition to 'year' (int), month (int), day(int) and hour(s), if provided. Other columns depend on what type of data is read.

    Raises
    ------
    IllegalArgumentException
        Year, month, day, or path to dataset wrong, so function cannot return any data.

    '''
    path = os.path.join(path_to_dataset, 'herring_catches_' + str(year) + '.csv')
    data = pd.read_csv(path, on_bad_lines='skip', encoding='utf8')
    month_data = data.loc[data['month'] == month]
    if day == None: return month_data
        
    if isinstance(day, int):
        day_data = month_data.loc[month_data['day'] == day]
    else:
        day_data = month_data.loc[month_data['day'].isin(day)]
    return day_data
    

def get_relevant_data_nc(path_to_dataset, variables, year, month, day, hour = None, DEPTH_LAYERS=15):
    '''
    A function to filter out the relevant data from a .nc file, in order to match them to catch data.

    Parameters
    ----------
    path_to_dataset : str
        Path to the dataset that is to be filtered. Has to be path to a folder of .nc files on the format: 'samples_year_month_date.nc'
    variables: list[str]
        List of strings that correspond to variables in nc file
    year : int
        Year of data to acquire. 
    month : int in [1,12]
        Month of data to acquire. 
    day : int in [1,31]
        Day of data to acquire.  
    hour(s): int(s) in [0,23], or a list of hours
        Hour(s) of data to acquire. If not provided, function will aquire all data from day. 
    Returns
    -------
       data : np.array DEPTH_LAYERSxlen(variables)x620x941

    '''
    XC, YC = 620, 941
    month = '0'+str(month) if month < 10 else str(month)
    day = '0'+str(day) if day < 10 else str(day)
    filename = 'samples_'+ str(year) + '.' + str(month) + '.' + str(day) + '_nonoverlap.nc'


    try:
        nc = Dataset(os.path.join(path_to_dataset, filename))
        hour = range(0,24) if hour == None else hour

        array_shape = (len(variables),1 if isinstance(hour, int) else len(hour), DEPTH_LAYERS,620, 941 )
        data =np.zeros(array_shape)
        for i, var in enumerate(variables):
            data[i,:,:,:,:] = nc[var][hour, 0:DEPTH_LAYERS,:,:]

        return data
    except Exception as e:
        print(e)
        print('Dataset not found. Check path and/or date')
        return None
    
    
    


if __name__ == '__main__': 
    #pred = np.load('test_prediction', allow_pickle=True)
    #print_on_map(pred)
    #plt.show()
    variables = ['temperature','u_east', 'v_north', 'salinity'] #, 'w_north', 'w_east,', 'w_velocity']
    data = get_relevant_data_nc('nor4km_data',variables,2020, 1, 1, 12) #PROBLEM: DATA ALWAYS MISSING
    Y = get_relevant_catch_data_csv('cleaned_datasets', 2020,1,5 )
    predictive_oceanographic_variables = []
    print(data.shape)
    for index, catch in Y.iterrows():
        x, y = ll2xy(catch['latitude'], catch['longitude'])
        print(data[:,0,0, y,x])
        predictive_oceanographic_variables.append(data[:,0,0, y,x])
    print(np.array(predictive_oceanographic_variables))
    pred = Y['Rundvekt'].to_numpy()
    print(pred)
    print_on_map(data[0,0,0,:,:], catch_data=Y)
    plt.show()
    model = LinearRegression()
    model.fit(predictive_oceanographic_variables, pred)
    r_sq = model.score(predictive_oceanographic_variables, pred)
    print(f"coefficient of determination: {r_sq}")
    print(f"slope: {model.coef_}")
    y_pred = model.predict(predictive_oceanographic_variables)
    print(f"predicted response:\n{y_pred}")

