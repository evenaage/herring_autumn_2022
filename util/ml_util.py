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
import sklearn
import h2o
from h2o.automl import H2OAutoML
import traceback

#these grid cell coordinates denote the borders of the norwegian economic zone
XC_LOW = 41 #41
YC_LOW = 157 # 157
XC_HIGH = 671# 671
YC_HIGH = 373 #373 #273 for norwegian economic zone?

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
    EARTHRAD = 6370
    GridData = [812, 638.5000, 4, 58, 60, 1]
    RAD = np.pi/180
    AN = EARTHRAD*(1+np.sin(abs(GridData[4])*RAD))/GridData[2]
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


def print_on_map(y,  catch_data = None,  figsize = (10,10), show_depth=None, 
ax = None, title="Herring distribution probability", cb_label = "Probability 0-1", normalize=True):
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
    if ax == None: 
        fig = plt.figure()
        fig.set_size_inches(figsize[0] , figsize[1])
    map = Basemap(projection='merc', llcrnrlon=-5.,llcrnrlat=56.,urcrnrlon=37.,urcrnrlat=75.,resolution='i', ax=ax ) #37, 75
    
    #prediction y is either a tensor or a numpy array, and needs to be a numpy array
    if type(y) == torch.tensor:
        pred_numpy= y.detach().numpy()[0,:,:]
    else:
        pred_numpy = y

    if normalize:
        max = np.max(pred_numpy)
        print(max)
        pred_normalized = pred_numpy/np.max(pred_numpy)
    else:
        pred_normalized = pred_numpy

    #
    #probabilities = map.contourf(lons[YC_LOW:YC_HIGH, XC_LOW:XC_HIGH], lats[YC_LOW:YC_HIGH, XC_LOW:XC_HIGH], pred_normalized[YC_LOW:YC_HIGH, XC_LOW:XC_HIGH],100, latlon=True, cmap =plt.cm.RdYlBu_r)
    probabilities = map.contourf(lons[YC_LOW:YC_HIGH, XC_LOW:XC_HIGH], lats[YC_LOW:YC_HIGH, XC_LOW:XC_HIGH],
        np.ma.masked_array(pred_normalized[YC_LOW:YC_HIGH, XC_LOW:XC_HIGH], mask[YC_LOW:YC_HIGH, XC_LOW:XC_HIGH]),100,\
           latlon=True, cmap =plt.cm.RdYlBu_r)
    cb = map.colorbar(probabilities,"bottom", size="5%", pad="2%", label=cb_label)

    #plot catch data, if possible.
    try:  
        map.scatter( catch_data['longitude'], catch_data['latitude'],latlon=True, marker='^')
    except Exception as e:
        #print("scatter didnt work")
        #print(repr(e))
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
                data[i,...] = nc[var][hour, 0:DEPTH_LAYERS,:,:]
                #print(nc[var][hour, 0:DEPTH_LAYERS,:,:].shape)
            except:
                d = nc[var][hour, :,:]
                #print(d.shape)
                d = np.expand_dims(d, 0)
                #print(d.shape)
                data[i,...] = d #this data des not have depth layers, e.g. wind
        if take_avg_of_depth_layers :data = average_depth_data(data, 2)[:,0,:,:]
        return data 
    except Exception:
        traceback.print_exc()
        print('Dataset not found. Check path and/or date')
        return None

def localize_plankton_file(path_to_dataset, year, month, day):
    month = '0'+str(month) if month < 10 else str(month)
    day = '0'+str(day) if day < 10 else str(day)
    filename = 'samplesb_'+ str(year) + '.' + str(month) + '.' + str(day) + '_nonoverlap.nc'
    if os.path.exists(os.path.join(path_to_dataset, filename)):
        return os.path.join(path_to_dataset, filename)
    else:
        filename = 'samplesb_'+ str(year) + '.' + str(month) + '.' + str(day) + '.nc'
        return os.path.join(path_to_dataset, filename)

def localize_nc_file(path_to_dataset, year, month, day):
    
    month = '0'+str(month) if month < 10 else str(month)
    day = '0'+str(day) if day < 10 else str(day)
    filename = 'samples_'+ str(year) + '.' + str(month) + '.' + str(day) + '_nonoverlap.nc'
    if os.path.exists(os.path.join(path_to_dataset, filename)):
        return os.path.join(path_to_dataset, filename)
    else:
        filename = 'samples_'+ str(year) + '.' + str(month) + '.' + str(day) + '.nc'
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


def calculate_gradient(data, index1, index0):
    gradients=[]
    variables = data.shape[0]
    #print(variables)
    for i in range(variables):
        gradient = np.gradient(data[...,i,index1 -1 : index1 +2,index0 -1: index0 +2 ])
        #print('data and gradients')
        #print(data[...,i,index1 -1 : index1 +2,index0 -1: index0 +2 ])
        #print(gradient)
        if gradient[0].mask.any() == True or gradient[1].mask.any() == True: return None
        gradients.append(np.sqrt(gradient[0].data[1,1]**2 + gradient[1].data[1,1]**2))
    return gradients

def find_data(data, index, calculate_gradients=False):
    '''
    Function to find data from the "closest" grid cell in case data is missing.

    '''
    FILL_VALUE = -32768
    def circle_around(data, index, step_length): #circle around grid cell, with greater step length to hit new ells
    
        for i in range(1, step_length+1):
            index[1] -= 1 #walk down
            values = list(data[...,index[1], index[0]])
            if calculate_gradients:
                gradients = calculate_gradient(data, index[1], index[0])
                if gradients == None : continue
                values += gradients
                if FILL_VALUE not in values:
                    return values
            else:
                if FILL_VALUE not in values:
                    return values
        for i in range(1, step_length+1):
            index[0] +=1 #walk right
            values = list(data[...,index[1], index[0]])
            if calculate_gradients:
                gradients = calculate_gradient(data, index[1], index[0])
                if gradients == None : continue
                values += gradients
            if FILL_VALUE not in values:
                    return values
            else:
                if FILL_VALUE not in values:
                    return values
        for i in range(1, step_length+1):
            index[1] += 1 #walk up
            values = list(data[...,index[1], index[0]])
            if calculate_gradients:
                gradients = calculate_gradient(data, index[1], index[0])
                if gradients == None : continue
                values += gradients
            if FILL_VALUE not in values:
                    return values
            else:
                if FILL_VALUE not in values:
                    return values
        for i in range(1, step_length+1):
            index[0] -= 1 #walk left
            values = list(data[...,index[1], index[0]])
            if calculate_gradients:
                gradients = calculate_gradient(data, index[1], index[0])
                if gradients == None : continue
                values += gradients
            if FILL_VALUE not in values:
                    return values
            else:
                if FILL_VALUE not in values:
                    return values
        index[0] -= 1
        index[1] +=1 
        return circle_around(data, index, 2*step_length) #double step length to circle around


    values = list(data[...,index[1], index[0]])

    if calculate_gradients: 
        gradients = calculate_gradient(data, index[1], index[0])
        if gradients != None: values += gradients
    if FILL_VALUE in values or ( calculate_gradients and gradients == None): 
        return circle_around(data, list((index[0] -1, index[1] +1)), step_length=2)
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
    tidspunkter = tracks['Tidspunkt (UTC)']
    tracks['month'] = [int(tid.split(' ')[0].split('.')[1]) for tid in tidspunkter]
    tracks['day'] = [int(tid.split(' ')[0].split('.')[0]) for tid in tidspunkter]
    tracks['hour'] = [int(tid.split()[1].split(':')[0]) for tid in tidspunkter]
    tracks['longitude'] = pd.to_numeric(tracks['Breddegrad'])
    tracks['latitude'] = pd.to_numeric(tracks['Lengdegrad'])
    return tracks



def get_catches_per_hour(catches_boat_day):
    catch_hours = []
    catch_per_hour = {}

    for idx, _catch in catches_boat_day.iterrows():
        catch_start = _catch['fangststart']
        catch_stop = _catch['fangstslutt']
        catch_hours = range(catch_start, catch_stop+1)

        catch_duration_hours = catch_stop - catch_start +1
        catch_amount = catches_boat_day['Rundvekt'] / catch_duration_hours #add this to make data a bit smaller
        for hour in catch_hours:
            if hour in catch_per_hour.keys(): catch_per_hour[hour] += catch_amount
            else: catch_per_hour[hour] = catch_amount
    return catch_hours, catch_per_hour


def read_and_store_dataset(year, month, variables, filename_ocean_data, filename_plankton, filename_target, *, replace = False, use_closest_grid_cell= True):
    '''
    God help me this function is long
    
    
    '''
    YEAR = year
    VARIABLES = variables
    MONTH = month
    vessel_dataframe = pd.read_csv('vessel_data\VMS_' + str(YEAR)+'.csv',delimiter=';', on_bad_lines='skip')
    if not replace:
        if os.path.exists(filename_plankton) or os.path.exists(filename_ocean_data) or os.path.exists(filename_target):
            print('file already exists for year: ', year)
            return None

    predictive_oceanographic_variables = []
    target = []
    catches = []
    plankton_data = []
    #in this loop, we stack both positive and assumed negative observations of herring in data, so we can train on them
    for day in range(0,32): #get data from days

        print('-------------------------- Day:', day, '------------------------------------------------')
        try:
            nc = Dataset(localize_nc_file('nor4km_data',YEAR, MONTH,day ))
            Y = get_relevant_catch_data_csv('cleaned_datasets', YEAR,MONTH,day )
            plankton = Dataset(localize_plankton_file('plankton_data', YEAR, MONTH,day))
        except:
            print("Found no data for day", day, " skipping.")
            continue
        
        data_added = {} #dict for remembering if we have already added negative track data for a vessel on a given day
        for index, catch in Y.iterrows():
            catches.append(catch)
            regnr = catch['Registreringsmerke']
            catches_boat = Y[Y['Registreringsmerke'] == regnr]
            catches_boat_day = catches_boat[catches_boat['day'] == day]

            #print(type(catch['Rundvekt']))


            vessel_data = get_vessel_track_data(vessel_dataframe,regnr,MONTH, day)

            #if we dont have vessel track data, just add catches            
            if not isinstance(vessel_data,pd.DataFrame): # or vessel_data.empty: 
                hour = catch['fangststart']
                data = get_relevant_data_nc(nc, VARIABLES,hour, take_avg_of_depth_layers=True)
                try:
                    x,y = ll2xy( catch['longitude'], catch['latitude']) #get x and y coord from lat and long
                except:
                    print('couldnt get x, y coord')
                    continue
                if use_closest_grid_cell:
                    d_plankton = find_data(plankton['Calanus_finmarchicus'][:,:,:],(x,y), True)
                    d = find_data(data[:,:,:], (x,y), True)
                else:
                    try:
                        d_plankton = list(plankton['Calanus_finmarchicus'][:,y,x] )
                        d = list(data[:,y,x])
                        d_plankton += calculate_gradient(plankton['Calanus_finmarchicus'], y, x)
                        d += calculate_gradient(data[:,:,:],y,x)
                    except Exception as e:
                        #print(traceback(e))
                        print('bad data, skipping')
                        continue
                plankton_data.append(d_plankton)
                predictive_oceanographic_variables.append(d) #add the oceanic data from the sqaures the vessel was in at the time
                catches.append(catch['Rundvekt'])
                continue

            #check if we have added the vessel track data for this ship from this day already. If so, we have also added the catch data.
            #Therefore, skip. This must ve dont after the code above, as if not we do not get catch data for multiple catches in days where there are no vessel data.
            if (regnr, day) in  data_added.keys(): continue
            data_added[(regnr, day)] = True
            
            catch_hours, catch_per_hour = get_catches_per_hour(catches_boat_day)
            last_catch = catch_hours[-1] #remove all hours after last catch, assume they are heading home
       
            #print('last_catch', last_catch)

            #add oceanic data for catch and non-catch hours. This is not independent data, maybe the day is over, maybe \
            # the boat is full, maybe they are going out to fish and not looking at echosounder. WIll include anyway,
            # as its better than no data.
            print('Day:' , day, ' Adding hours:')
            added_this_hour = {}
            '''if vessel_data.empty: continue
            first_hour = vessel_data.iloc[0]['hour']
            last_hour = vessel_data.iloc[-1]['hour']
            print(slice(first_hour,last_hour))'''
            for index, row in vessel_data.iterrows():
                hour = row['hour']
                if hour > last_catch: 
                    print("our of last catch reached, ignore data")
                    break
                if hour in added_this_hour.keys(): continue # or hour > last_catch: continue
                added_this_hour[hour] = True
                print(regnr, hour)
                data = get_relevant_data_nc(nc, VARIABLES,hour,take_avg_of_depth_layers=True) #get month/day/hour data
                try:
                    x,y = ll2xy( row['longitude'], row['latitude']) #get x and y coord from lat and long
                    #print(row['longitude'], row['latitude'],x,y)
                except:
                    print("couldnt get x, y coord")
                    continue
                if use_closest_grid_cell:
                    d_plankton = find_data(plankton['Calanus_finmarchicus'][:,:,:],(x,y), True)
                    d = find_data(data[:,:,:], (x,y), True)
                else:
                    try:
                        d_plankton = list(plankton['Calanus_finmarchicus'][:,y,x] )
                        d = list(data[:,y,x])
                        print(d)
                        d_plankton += calculate_gradient(plankton['Calanus_finmarchicus'], y, x)
                        grad = calculate_gradient(data,y,x)
                        #print('gradients: ', grad, d_plankton)
                        #print(grad, calculate_gradient(plankton['Calanus_finmarchicus'], y, x))
                        d += calculate_gradient(data[:,:,:],y,x)
                        
                    except Exception as e:
                        #traceback.print_exception(e)
                        print('bad data, skipping')
                        continue
                print('data: ', d, 'plankton data:' , d_plankton)
                plankton_data.append(d_plankton)
                predictive_oceanographic_variables.append(d) #add the oceanic data from the sqaures the vessel was in at the time
                #print(d, d_plankton)
                if hour in catch_hours:     #add catch. actual catch or 0
                    target.append(catch_per_hour[hour]) 
                else:
                    target.append(0) 

    #print(np.array(plankton_data).shape)
    if len(predictive_oceanographic_variables) == 0 :
        print("no data for ", year, " and month: ", month)
        return None
    predictive_oceanographic_variables = np.array(predictive_oceanographic_variables)
    #predictive_oceanographic_variables = np.append(predictive_oceanographic_variables, plankton_data,axis=1)
    #print(predictive_oceanographic_variables.shape)
    plankton_data = np.array(plankton_data)
    target = np.array(target)

    
    np.save(filename_ocean_data, predictive_oceanographic_variables)
    np.save(filename_plankton, plankton_data)
    np.save(filename_target, target)

def load_and_combine_data(year, month,folder,  skip_bad_cells=False):
    if month == 'jan':
        data  = np.load(os.path.join(folder, 'january_' + str(year)+'_ocean_data.npy'), allow_pickle=True)
        plankton = np.load(os.path.join(folder,'plankton' +'_january_'+str(year) + '.npy'), allow_pickle=True )
        target = np.load(os.path.join(folder,'target_january_' + str(year)+ '.npy'), allow_pickle=True)
    elif month == 'dec':
        data  = np.load(os.path.join(folder,'december_'+  str(year)+'_ocean_data.npy'), allow_pickle=True)
        plankton = np.load(os.path.join(folder,'plankton_'   + 'december_'  +str(year)+ '.npy'), allow_pickle=True )
        target = np.load(os.path.join(folder,'target_december_' +  str(year)+ '.npy'), allow_pickle=True)
    data = np.append(data, plankton, axis=1)
    target = np.array([target[i].iloc[0] if isinstance(target[i], pd.Series) else target[i] for i in range(len(target))])

    return data, target

def predict_per_grid_cell(model, ocean_data):
    '''
    Give a grid cell model and predict herring distribution based on ocean data
    '''
    relevant_data = ocean_data[:,YC_LOW:YC_HIGH,XC_LOW:XC_HIGH]
    predictions = model(relevant_data)
    return predictions


def validate(model, prediction, truth, cutoff):
    '''
    Validates a prediciton based on different metrics
    '''
    names = ['Temperature',  'Salinity', 'u_east', 'v_north',  'w_north', 'w_east','Temperature gradient', \
     'Salinity gradient','w_east_gradient', 'u_east gradient','v_north_gradient','w_north_gradient','Calanus finmarchicus', \
        'Calanus finmarchicus gradient', 'Catch']
    #fig, axs = plt.subplots(2,2)
    #fig.set_size_inches(15,15)
    VARIABLES = ['temperature','salinity','u_east', 'v_north',  'w_north', 'w_east']
    nc = Dataset(localize_nc_file('nor4km_data',2021, 1,20 ))
    data = get_relevant_data_nc(nc,VARIABLES,12)
    mask = data[0,...].mask
    prediction = [1 if p > cutoff else 0 for p in prediction]
    #print(prediction.shape, truth.shape)
    acc = (prediction == truth)
    #print(acc.shape)
    sens = 0
    spec = 0
    pos = 0
    neg = 0
    for i in range(len(acc)):
        if acc[i] == True:
            if prediction[i] == 1:
                pos += 1
                sens += 1
            else:
                spec += 1
                neg += 1
        else:
            if prediction[i] == 1:
                pos += 1
            else:
                neg += 1

    #spec = sum([0 if (acc[i] == False and prediction[i]) == 0 else 0 for i in range(len(truth))])/ (len(truth) - np.count_nonzero(truth))
    acc = np.count_nonzero(prediction == truth) / len(truth)
    sens = sens/pos
    spec = spec/neg
    plankton = Dataset(localize_plankton_file('plankton_data', 2021, 1, 20))['Calanus_finmarchicus']
    #print(data.shape, plankton.shape)
    data = np.append(data, plankton, axis=0)
    grads = []
    for i in range(len(VARIABLES) +1):
        grad = np.gradient(data[i,:,:])
        grad = np.sqrt(np.square(grad[0].data) + np.square(grad[0].data))
        grads.append(grad)
    #print(data.shape, np.array(grads).shape)
    data = np.append(data, grads, axis = 0)
    #print(data[0,:,:])
    #print_on_map(data[0,:,:])
    print(data.shape)
    #data = np.reshape(data, (620*941, 14))
    print(data.shape)
    #func = np.vectorize(model)
    print(data[:,...].shape, data.shape)
    pred = np.zeros((620, 941))
    predictions  =[]
    for xc in range(XC_LOW, XC_HIGH):
        if xc % 100 ==0: print(xc)
        for yc in range(YC_LOW, YC_HIGH):
            try:
                val =model(torch.tensor(data[:,yc,xc], dtype=torch.float32)).detach().numpy() 
                pred[yc, xc] = val
                predictions.append(val)
            except:
                #print(model.predict(h2o.H2OFrame(data[:,yc,xc], column_names = names)).as_data_frame().to_numpy())
                data = np.reshape(data, (620*941, 14))
                pred[yc, xc] = np.mean(model.predict(h2o.H2OFrame(data[:,yc,xc], column_names = names)).as_data_frame().to_numpy()[:,0])

    #[[pred[i,j] = model(torch.tensor(data[:,i,j], dtype=torch.float32)).detach().numpy() for i in range(XC_LOW,XC_HIGH)] for j in range(YC_LOW, YC_HIGH)]
    #print(pred)
    #print(pred[pred > 0])
    #pred = np.array(list(map(model,torch.tensor(data, dtype=torch.float32))))
    #try: pred = model(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    #except: pred = model.predict(h2o.H2OFrame(data, column_names = names)).as_data_frame().to_numpy()[:,0]
   
    data = np.reshape(data, (14, 620, 941))
    plt.hist(np.array(predictions),bins=20)
    plt.show()
    #print(pred[...,0].shape)
    #print(np.transpose(pred[...,0]).shape)
    #pred = np.transpose(pred[...,0])
    #pred = np.reshape(pred, ( 1,620,941))
    print("Non zero predictions in norwegian economic zone:" , np.count_nonzero(pred[YC_LOW:YC_HIGH,XC_LOW:XC_HIGH]))
    print(pred.shape)
    print('Sensitivity: ', sens, ' Specificity: ', spec,' Accuracy: ', acc)
    catch_data = get_relevant_catch_data_csv('cleaned_datasets', 2021,1,20 )
    print_on_map(np.ma.masked_array(pred,mask) , catch_data,normalize=False)
    '''
    axs[0,0].set_title('Herring prediction')
    print_on_map(data[0,...], ax = axs[1,0], cb_label = 'Temperature low to high', normalize=False)
    axs[1,0].set_title('Temperature')
    print_on_map(data[-2,...], ax = axs[0,1], cb_label = "Plankton density low to high",normalize=False)
    axs[0,1].set_title('Plankton distribution')
    axs[1,1].set_title('Prediction histogram')
    #,ax = axs[1,1])
    '''
    plt.show()





if __name__ == '__main__': 
    #fig, (ax1, ax2) = plt.subplots(1,2)
    #temperature = Dataset(localize_nc_file('nor4km_data', 2021, 1, 5))['temperature'][12,0,:,:]
    #temperature = np.reshape(temperature, (1,-1))
    #temperature = np.reshape(temperature, ((620, 941)))
    #print_on_map(temperature, catch_data=get_relevant_catch_data_csv('cleaned_datasets', 2021, 1, 1 ))
    #print_on_map(temperature, catch_data=get_relevant_catch_data_csv('cleaned_datasets', 2021, 1, 1 ))

    #plt.show()
    '''
    data = get_relevant_data_nc(nc, VARIABLES,0)
    grads = []
    for i in range(len(VARIABLES)):
        grad = np.gradient(data[i,:,:])
        grad = np.sqrt(np.square(grad[0].data) + np.square(grad[0].data))
        grads.append(grad)
    

    data = np.append(data, grads, axis = 0)
    plankton = Dataset(localize_plankton_file('plankton_data', 2021, JANUARY,28))
    grad = np.gradient[plankton['Calanus_finmarchicus'][0,:,:]]
    grad = np.sqrt(np.square(grad[0].data) + np.square(grad[0].data))
    plankton = np.append(plankton, grad, axis=0)
    data = np.append(data, plankton, axis=0)
    reshaped = np.reshape(data, ( 620*941,2*len(VARIABLES) + 2))
    print(reshaped.shape)



    y_pred = model.predict(reshaped)
    print(y_pred.shape)
    y_pred = np.reshape(y_pred, (620,941))
    '''
    #temperature = Dataset(localize_nc_file('nor4km_data', 2021, 1, 5))['temperature'][12,0,:,:]
    #print_on_map(temperature, catch_data=get_relevant_catch_data_csv('cleaned_datasets', 2021, 1, 1 ))
    #print(f"predicted response:\n{y_pred}")
    #plt.show()
    folder = 'dataset_2_skip_bad_cells'
    VARIABLES = ['temperature','salinity','u_east', 'v_north',  'w_north', 'w_east']
    #read_and_store_dataset(2021, 1,VARIABLES, '6_vars_+_plankton_15_depth_layers_january2021', 'target_jan_2021' )
    #read_and_store_dataset(2020, 1,VARIABLES, '6_vars_+_plankton_15_depth_layers_january2020', 'target_jan_2020' )
    for year in range(2019, 2022):
        read_and_store_dataset(year, 1, VARIABLES, os.path.join(folder, 'january_' + str(year)+'_ocean_data'), \
        os.path.join(folder, 'plankton_january_'+str(year)) ,os.path.join(folder, 'target_december_' + str(year)),use_closest_grid_cell=False)
    for year in range(2019, 2022):
        read_and_store_dataset(year, 1, VARIABLES, os.path.join(folder, 'december_' + str(year)+'_ocean_data'), \
            os.path.join(folder, 'plankton_december_'+str(year)) ,os.path.join(folder, 'target_december_' + str(year)),use_closest_grid_cell=False)
    
