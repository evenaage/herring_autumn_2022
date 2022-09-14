import pandas as pd
import geopandas as gpd
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.express as px

fig, ax = plt.subplots()
catch_data = pd.read_csv('fangstdata\elektronisk-rapportering-ers-2018-fangstmelding-dca.csv',delimiter=';', on_bad_lines='skip', encoding='utf8')
#print(catch_data.columns)

countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
herring_catches = catch_data.loc[catch_data['Art - FDIR'] == 'Sild']
#herring_catches.loc['Startposisjon bredde'] = pd.to_numeric([pos.replace(',','.') for pos in herring_catches['Startposisjon bredde']])
#herring_catches.loc['Startposisjon lengde'] = pd.to_numeric([pos.replace(',','.') for pos in herring_catches['Startposisjon lengde']])
longitude = pd.to_numeric([pos.replace(',','.') for pos in herring_catches['Startposisjon bredde']])
latitude = pd.to_numeric([pos.replace(',','.') for pos in herring_catches['Startposisjon lengde']])

#herring_catches[:1000].plot.scatter(x='Startposisjon lengde', y='Startposisjon bredde', xlim=[0,90], ylim = [0,90], ax=ax)
#global plot
#plot = plt.scatter(latitude, longitude, marker='.')
#print(min(herring_catches['Startposisjon bredde']), max(herring_catches['Startposisjon bredde']))
#print(min(herring_catches['Startposisjon lengde']), max(herring_catches['Startposisjon lengde']))
plt.subplots_adjust(left=0.25, bottom=0.25)

#plt.xlim([0,30])
#plt.ylim([50, 83])
#slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
#month_slider = Slider(ax = slider_ax, label = 'Month',valmin=1, valmax=12, valinit=1, valstep=[i for i in range(1,13)])
#ax.grid(visible=True, alpha=.5)
#print(herring_catches.columns[51],herring_catches.columns[83],herring_catches.columns[90])
#plt.sca(ax)
def update(val):
    longitude, latitude = [],[]
    for index, row in herring_catches.iterrows():
        if int(row['Startdato'].split('.')[1]) == val:
            longitude.append(row['Startposisjon bredde'])
            latitude.append(row['Startposisjon lengde'])
    longitude = pd.to_numeric([pos.replace(',','.') for pos in longitude])
    latitude = pd.to_numeric([pos.replace(',','.') for pos in latitude])
    plt.scatter(latitude, longitude, marker='.', alpha=.3)

#for i in range(1,13):
#    update(i)

#month_slider.on_changed(update)
plotter = {'latitude' : latitude, 'longitude': longitude}

fig = px.density_mapbox(plotter, lat='longitude', lon='latitude',mapbox_style='stamen-terrain', opacity=.7, radius=5, title="Sildefangst 2018")
fig.show()

#plt.show()