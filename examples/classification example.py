#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd

sys.path.append('../')
from StopGoClassifier import StopGoClassifier



# read EPSG 3035 projected GPS samples
gps_data = pd.read_csv('data/preprocessed_example.csv')

# convert timestamps from string to datetime64[ns, Europe/Berlin] objects
gps_data.ts = pd.to_datetime(gps_data['ts'], utc=True)
gps_data = gps_data.set_index('ts').tz_convert('Europe/Berlin').reset_index()



# create classifier object
classifier = StopGoClassifier()

# initiate classfier with GPS samples (motion score is optional)
classifier.read(gps_data.ts, gps_data.x, gps_data.y, gps_data.motion_score)

# run the classifier
# this invokes process_samples(), aggregate(), filter_outliers(), & isolate_trip_samples()
# it returns a list of stop intervals
stop_intervals = classifier.run()
print(stop_intervals)