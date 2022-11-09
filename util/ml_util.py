from turtle import circle
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
            The acquired data. Columns in dataframe include: 'Registreringsmerke', 'latitude', 'longitude', 'rundvekt' and 'start_date' (str) and 'end_date', in addition to 'year' (int), month (int), day(int) and hour(s), if provided. Other columns depend on what type of data is read.

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
    

def get_relevant_data_nc(dataset, variables, hour = None, take_avg_of_depth_layers=True, DEPTH_LAYERS=15):
    '''
    A function to filter out the relevant data from a .nc file, in order to match them to catch data.

    Parameters
    ----------
    dataset : .nc file
        Dataset to sift through
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

    try:
        nc = dataset
        hour = range(0,24) if hour == None else hour
        #print(nc['gridLats'][0][0], nc['gridLons'][0][0],nc['gridLats'][-1][-1], nc['gridLons'][-1][-1])
        array_shape = (len(variables),1 if isinstance(hour, int) else len(hour), DEPTH_LAYERS,620, 941 )
        data =np.zeros(array_shape)
        for i, var in enumerate(variables):
            try:
                data[i,:,:,:,:] = nc[var][hour, 0:DEPTH_LAYERS,:,:]
                #print(nc[var][hour, 0:DEPTH_LAYERS,:,:].shape)
            except:
                d = nc[var][hour, :,:]
                #print(d.shape)
                d = np.expand_dims(d, 0)
                #print(d.shape)
                data[i,:,:,:,:] = d #this data des not have depth layers, e.g. wind
        if take_avg_of_depth_layers :data = average_depth_data(data, 2)
        return data[:,0,:,:] #remove depth data, that layer is only 1 var and therefore superflous
    except Exception as e:
        print(e)
        print('Dataset not found. Check path and/or date')
        return None

def localize_nc_file(path_to_dataset, year, month, day):
    
    month = '0'+str(month) if month < 10 else str(month)
    day = '0'+str(day) if day < 10 else str(day)
    filename = 'samples_'+ str(year) + '.' + str(month) + '.' + str(day) + '_nonoverlap.nc'
    return os.path.join(path_to_dataset, filename)
    
def average_depth_data(data, depth_layer_index):
    '''
    Averages depth layers to reduce dimension of input data.

    Parameters
    ----------
    data: np.array
        Array of data
    depth_layer_index: int
        index of depth layers

    Returns
    -------
    mean_data: np.array
        array where depth data is averaged
    '''
    FILL_VALUE = -32768
    masked =np.ma.masked_values(data, FILL_VALUE)
    mean_data = np.mean(masked, depth_layer_index)
    return mean_data


def find_data(data, index):
    '''
    Function to find data from the "closest" grid cell in case data is missing.

    '''
    FILL_VALUE = -32768

    def circle_around(data, index, step_length): #circle around grid cell, with greater step length to hit new ells
        #print(index)
        for i in range(1, step_length+1):
            index[1] -= 1 #walk down
            values = data[:,index[1], index[0]]
            if FILL_VALUE not in values:
                return values
        for i in range(1, step_length+1):
            index[0] +=1 #walk right
            values = data[:,index[1], index[0]]
            if FILL_VALUE not in values:
                return values
        for i in range(1, step_length+1):
            index[1] += 1 #walk up
            values = data[:,index[1], index[0]]
            if FILL_VALUE not in values:
                return values
        for i in range(1, step_length+1):
            index[0] -= 1 #walk left
            values = data[:,index[1], index[0]]
            if FILL_VALUE not in values:
                return values
        index[0] -= 1
        index[1] +=1 
        return circle_around(data, index, 2*step_length) #double step length to circle around

    

    values = data[:,index[1], index[0]]
    if FILL_VALUE in values: return circle_around(data, list((index[0] -1, index[1] +1)), step_length=2)
    else: return values


def get_vessel_track_data(df, reg, måned, dag):
    '''
    A function to filter out the relevant vessel track data and return them as a df. Does not care wether fish are caught or not.

    Parameters
    ----------
    df : pd.DataFrame
        a pandas dataframe with multiple vessel track data for a specific year
    reg: str
        registration number of vessel to track (Registreringsmerke)
    måned: int
        month of which we want vessel data
    dag: int
        day of which we want vessel data
    
    Returns
    -------
    df: pd.DataFrame
        A dataframe containing the vessel data for the specific day. Also adds columns 'month', 'day', and 'hour', in 
        addition to translate lons and lats to numeric.
    '''
    tracks = df[df['Registreringsmerke'] == reg]
    tidspunkter = tracks['Tidspunkt (UTC)']
    tracks_correct_day = []
    #for tid in tidspunkter:
    tracks_correct_day =[int(tid.split()[0].split('.')[0]) == dag \
        and int(tid.split(' ')[0].split('.')[1]) == måned for tid in tidspunkter]
    #print(tracks_correct_day)
    if len(tracks_correct_day) == 0:
        return None
    tracks = tracks.iloc[tracks_correct_day]
    #print(tracks)
    tidspunkter = tracks['Tidspunkt (UTC)']
    tracks['month'] = [int(tid.split(' ')[0].split('.')[1]) for tid in tidspunkter]
    tracks['day'] = [int(tid.split(' ')[0].split('.')[0]) for tid in tidspunkter]
    tracks['hour'] = [int(tid.split()[1].split(':')[0]) for tid in tidspunkter]
    tracks['longitude'] = pd.to_numeric(tracks['Breddegrad'])
    tracks['latitude'] = pd.to_numeric(tracks['Lengdegrad'])
    return tracks


    
#def validate(predicted, )


if __name__ == '__main__': 
    VARIABLES = ['temperature','u_east', 'v_north', 'salinity', 'w_north', 'w_east']
    JANUARY =1
    vessel_dataframe_2020 = pd.read_csv('vessel_data\VMS_2020.csv',delimiter=';', on_bad_lines='skip')


    #in this loop, we stack both positive and assumed negative observations of herring in data, so we can train on them
    for day in range(1,2): #get data from some of january
        print('Day:', day)
        nc = Dataset(localize_nc_file('nor4km_data',2020, JANUARY,day ))
        Y = get_relevant_catch_data_csv('cleaned_datasets', 2020,JANUARY,day )

        predictive_oceanographic_variables = []
        target = []
        catches = []
        data_added = {} #dict for remembering if we have already added negative track data for a vessel on a given day
        for index, catch in Y.iterrows():
            catches.append(catch)
            regnr = catch['Registreringsmerke']
            catches_boat = Y[Y['Registreringsmerke'] == regnr]
            catches_boat_day = catches_boat[catches_boat['day'] == day]


            vessel_data = get_vessel_track_data(vessel_dataframe_2020,regnr,JANUARY, day)

            #if we dont have vessel track data, just add catches            
            if not isinstance(vessel_data,pd.DataFrame): 
                hour =catch['fangststart']
                data = get_relevant_data_nc(nc, VARIABLES,hour)
                x,y = ll2xy( catch['longitude'], catch['latitude']) #get x and y coord from lat and long
                predictive_oceanographic_variables.append(find_data(data[:,:,:], (x,y))) #add the oceanic data from the sqaures the vessel was in at the time
                catches.append(catch['Rundvekt'])

            #check if we have added the vessel track data for this ship from this day already. If so, we have also added the catch data.
            #Therefore, skip. This must ve dont after the code above, as if not we do not get catch data for multiple catches in days where there are no vessel data.
            if (regnr, day) in  data_added.keys(): continue
            data_added[(regnr, day)] = True
            
            catch_hours = []
            catch_per_hour = {}
            for idx, _catch in catches_boat_day.iterrows():
                catch_start = _catch['fangststart']
                catch_stop = _catch['fangstslutt']
                catch_hours += range(catch_start, catch_stop+1)

                catch_duration_hours = catch_stop - catch_start +1
                catch_amount = catch['Rundvekt'] / catch_duration_hours #add this to make data a bit smaller
                for hour in catch_hours:
                    if hour in catch_per_hour.keys(): catch_per_hour[hour] += catch_amount
                    else: catch_per_hour[hour] = catch_amount


            #add oceanic data for catch and non-catch hours. This is not independent data, maybe the day is over, maybe \
            # the boat is full, maybe they are going out to fish and not looking at echosounder. WIll include anyway,
            # as its better than no data.
            print('Adding hours:')
            for index, row in vessel_data.iterrows():
                #print(row)
                hour = row['hour']
                print(regnr, hour)


                data = get_relevant_data_nc(nc, VARIABLES,hour) #get month/day/hour data
                #print(type(row['latitude']), row['longitude'])
                print("get_relevant_data ran")
                x,y = ll2xy( row['longitude'], row['latitude']) #get x and y coord from lat and long
                print("ll2xy ran")
                predictive_oceanographic_variables.append(find_data(data[:,:,:], (x,y))) #add the oceanic data from the sqaures the vessel was in at the time
                print('data added')
                if hour in catch_hours:     #add catch. actual catch or 0
                    target.append(catch_per_hour[hour]) 
                else:
                    target.append(0) 
        
            
    print(np.array(predictive_oceanographic_variables).shape)
    target = np.array(target)
    #print(pred)
    print_on_map(data[0,:,:], catch_data=catches)
    plt.show()
    model = LinearRegression()
    model.fit(predictive_oceanographic_variables, target)
    r_sq = model.score(predictive_oceanographic_variables, target)
    print(f"coefficient of determination: {r_sq}")
    print(f"slope: {model.coef_}")
    y_pred = model.predict(predictive_oceanographic_variables)
    print(f"predicted response:\n{y_pred}")

