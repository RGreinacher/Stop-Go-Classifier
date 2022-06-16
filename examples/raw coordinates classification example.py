#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import geopandas as gpd

sys.path.append('../')
from StopGoClassifier import StopGoClassifier



# read raw GPS samples (containing only timestamp, longitude & latitude information)
gps_data = pd.read_csv('data/raw_example.csv')

# convert timestamps from string to datetime64[ns, Europe/Berlin] objects
gps_data.ts = pd.to_datetime(gps_data['ts'], utc=True)
gps_data = gps_data.set_index('ts').tz_convert('Europe/Berlin').reset_index()

# project coordinates
geometry = gpd.points_from_xy(gps_data.latitude, gps_data.longitude)
gdf = gpd.GeoDataFrame(gps_data, geometry=geometry)
gdf = gdf.set_crs('EPSG:4326') # interpret raw GPS records as WGS84 / EPSG 4326
gdf = gdf.to_crs('EPSG:3035') # project sample into the ETRS89-extended / LAEA Europe space

# extract x / y values from geopandas geometry object
gdf['y'] = list(map(lambda x: x.x, gdf.geometry.values))
gdf['x'] = list(map(lambda x: x.y, gdf.geometry.values))

# create a new dataframe containing only relevant information
gps_data = gdf[['ts', 'x', 'y', 'longitude', 'latitude']].copy()



# create classifier object
classifier = StopGoClassifier()

# initiate classfier with GPS samples
classifier.read(gps_data.ts, gps_data.x, gps_data.y)

# run the classifier
stop_intervals = classifier.run()
print(stop_intervals)