#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
from StopGoClassifier import StopGoClassifier



# read EPSG 3035 projected GPS samples
gps_data = pd.read_csv('data/preprocessed_example.csv')
gps_data.ts = pd.to_datetime(gps_data['ts'], utc=True)
gps_data = gps_data.set_index('ts').tz_convert('Europe/Berlin').reset_index()



# plot all samples
fig = plt.figure(figsize=(10,10))
plt.xlabel('x projection')
plt.ylabel('y projection')
plt.title('EPSG 3035 projected GPS coordinates')
plt.plot(gps_data.x, gps_data.y, c=(0.3, 0.3, 0.3, 1.0), ls='--')
plt.scatter(gps_data.x, gps_data.y, s=30, color=(0.3, 0.3, 0.3, 1.0), marker='x')
plt.show()



# instantiate classifier & run classification
classifier = StopGoClassifier()
classifier.read(gps_data.ts, gps_data.x, gps_data.y, gps_data.motion_score)
stop_intervals = classifier.run()



# plot samples and highlight the dectected stops
fig = plt.figure(figsize=(10,10))
plt.xlabel('x projection')
plt.ylabel('y projection')
plt.title('Samples with detected stop clusters')
plt.plot(gps_data.x, gps_data.y, c=(0.3, 0.3, 0.3, 1.0), ls='--')

# plot dwell points; alternate colors to distinguish stops
for index, row in stop_intervals.iterrows():
    color = (0.9, 0.7, 0.3, 1.0) if index % 2 else (0.95, 0.45, 0.3, 1.0)
    row = stop_intervals.iloc[index]
    dwell_slice = gps_data[(gps_data.ts >= row.start) & (gps_data.ts <= row.stop)]

    plt.scatter(dwell_slice.x, dwell_slice.y, s=100, color=color, marker='x')
    plt.scatter(row.x, row.y, s=200, color=color, marker='o')

plt.show()