#!/usr/bin/env python3

from scipy import spatial
import numpy as np
import pandas as pd



class StopGoClassifier():

  def __init__(self, overwrite_settings=None):
    # init members
    self.original_df = None
    self.samples_df = pd.DataFrame()
    self.stop_df = pd.DataFrame()
    self.debug_stop_merge_df = None
    self.trip_df = None
    self.trip_samples_df = None
    self.settings = {
        'MIN_STOP_INTERVAL': 63, # time in seconds, stops below this threshold will be ignored
        'RELEVANT_STOP_DURATION': 178, # time in seconds, stops longer than this will always be kept
        
        'MIN_DISTANCE_BETWEEN_STOP': 37, # distance in meter; min distance two consecutive stop places must have
        'RELEVANT_DISTANCE_BETWEEN_STOP': 165, # distance in meter; stop with such a distance will always be kept
        'MIN_TIME_BETWEEN_STOPS': 69, # time in seconds; remove or merge if less than this threshold
        'RELEVANT_TIME_BETWEEN_STOPS': 131, # time in seconds; a trip between two stops is relevant if it is longer than this threshold
        'MAX_TIME_BETWEEN_STOPS_FOR_MERGE': 175, # time in seconds; should not merge stops having more than this time between each other

        # METHOD 1: Motion Score
        'USE_MOTION_SCORE': True, # ignores motion score if set to False
        'MOTION_SCORE_LOWER_CUTOFF': 0.29,
        'MOTION_SCORE_THRESHOLD': 1.30, # below 30: stop is likely / above: trip is likely
        'MOTION_SCORE_UPPER_CUTOFF': 3.00,

        # METHOD 2: rectangle_distance_ratio is always > 1 and usually < 4
        'USE_METHOD_RDR': True,
        'METHOD_RECTANGLE_DISTANCE_WINDOW_SIZE': 23,
        'METHOD_RECTANGLE_DISTANCE_RATIO_THRESHOLD': 1.95,
        'METHOD_RECTANGLE_DISTANCE_RATIO_UPPER_CUTOFF': 2.875,
        'METHOD_RECTANGLE_DISTANCE_RATIO_WEIGHT': 0.735,

        # METHOD 3: Bearing analysis
        'USE_METHOD_BA': True,
        'METHOD_BEARING_ANALYSIS_LOWER_CUTOFF': 31,
        'METHOD_BEARING_ANALYSIS_THRESHOLD': 41,
        'METHOD_BEARING_ANALYSIS_UPPER_CUTOFF': 82,
        'METHOD_BEARING_ANALYSIS_WINDOW_SIZE': 15,
        'METHOD_BEARING_ANALYSIS_WEIGHT': 1.2,

        # METHOD 4: Analysis of distance between path start and end
        'USE_METHOD_SEDA': True,
        'METHOD_START_END_DISTANCE_ANALYSIS_LOWER_CUTOFF': 19,
        'METHOD_START_END_DISTANCE_ANALYSIS_THRESHOLD': 95,
        'METHOD_START_END_DISTANCE_ANALYSIS_UPPER_CUTOFF': 262,
        'METHOD_START_END_DISTANCE_ANALYSIS_WINDOW_SIZE': 14,
        'METHOD_START_END_DISTANCE_ANALYSIS_WEIGHT': 1.125,

        # METHOD 5: Analysis of intersections of path segments
        'USE_METHOD_ISA': True,
        'METHOD_INTERSECTING_SEGMENTS_ANALYSIS_UPPER_CUTOFF': 4,
        'METHOD_INTERSECTING_SEGMENTS_ANALYSIS_THRESHOLD': 0.75,
        'METHOD_INTERSECTING_SEGMENTS_ANALYSIS_WINDOW_SIZE': 19,
        'METHOD_INTERSECTING_SEGMENTS_ANALYSIS_WEIGHT': 0.43,

        # METHOD 6: Analysis of data gaps
        'USE_METHOD_MDA': True,
        'MIN_MISSING_DATA_INTERVAL': 53, # time in seconds; if there are no records for at least this persiod of time between two consecutive samples, overwite scores using the missing data analysis strategy
        'METHOD_MISSING_DATA_ANALYSIS_LOWER_CUTOFF': 0.39, # max tolerated drift speed to have assured stop
        'METHOD_MISSING_DATA_ANALYSIS_THRESHOLD': 1.4,
        'METHOD_MISSING_DATA_ANALYSIS_UPPER_CUTOFF': 26, # min speed to have assured trip
      }

    if overwrite_settings is not None:
      self.settings.update(overwrite_settings)

    self.max_window_size = max([
      self.settings['METHOD_RECTANGLE_DISTANCE_WINDOW_SIZE'],
      self.settings['METHOD_BEARING_ANALYSIS_WINDOW_SIZE'],
      self.settings['METHOD_START_END_DISTANCE_ANALYSIS_WINDOW_SIZE'],
      self.settings['METHOD_INTERSECTING_SEGMENTS_ANALYSIS_WINDOW_SIZE'],
    ])

    # check if any method is selected
    check_flag = False
    for METHOD_setting in ['USE_METHOD_RDR', 'USE_METHOD_BA', 'USE_METHOD_SEDA']:
      check_flag |= self.settings[METHOD_setting]
    if not check_flag:
      raise Exception("Bad settings: no method selected")
  
  def read(self, timestamps, x_coordinates, y_coordinates, motion_score=None, index=None):
    df = pd.DataFrame()
    df['ts'] = timestamps
    df['x'] = x_coordinates
    df['y'] = y_coordinates
    df['motion_score'] = motion_score

    if index is not None:
      df['index'] = index
    else:
      df['index'] = range(0, df.shape[0])

    # force deactivate motion score analysis if no motion score values are available
    if motion_score is None:
      self.settings['USE_MOTION_SCORE'] = False

    # filter duplicates and drop NA rows
    df.drop_duplicates(subset=['x', 'y'], keep='first', inplace=True)
    df.dropna(subset=['x', 'y'], inplace=True)
    assert df.shape[0] >= self.max_window_size, f'Classification requires at least {self.max_window_size} unique samples'

    self.original_df = df
    return self

  def run(self):
    # classify each sample as stop or trip
    self.process_samples()
    
    # aggregate stops; writes into self.stop_df
    self.aggregate()

    if self.stop_df.shape[0] > 0:
      # filter and merge stops; overwrites self.stop_df
      self.filter_outliers()
    
    # isolate GPS records belonging to trips
    self.isolate_trip_samples()

    return self.stop_df



  #####################################################
  # main functions: run, process & aggregate
  #####################################################
  def process_samples(self):
    self.samples_df = self.original_df
    # add attributes
    self.samples_df = StopGoClassifier.add_attributes(self.samples_df, dist=True, bearing=True)
    if self.samples_df[self.samples_df.distance_to_next == 0].shape[0] > 0:
      self.samples_df = self.samples_df[self.samples_df.distance_to_next > 0].copy()
      self.samples_df = StopGoClassifier.add_attributes(self.samples_df, bearing=True)

    self.samples_df['key'] = self.samples_df.apply(lambda row: (row.x, row.y), axis=1)
    self.samples_df['id'] = range(self.samples_df.shape[0])
    self.samples_df['uncertain'] = True # initially, all samples states are uncertain
    self.samples_df['x_shifted'] = self.samples_df.x.shift(-1)
    self.samples_df['y_shifted'] = self.samples_df.y.shift(-1)

    # **************
    # Method no. 1 - use ACC score to determine assured stop
    if self.settings['USE_MOTION_SCORE']:
      # samples without any value are mapped to zero
      idx_without_ms = self.samples_df[self.samples_df.motion_score.isna()].index
      self.samples_df.loc[idx_without_ms, 'motion_score'] = 0.0
      
      # samples with motion score are mapped to the [-1, 1] interval; -1 = trip, 1 = stop
      idx_with_ms = self.samples_df[~self.samples_df.index.isin(idx_without_ms)].index
      self.samples_df.loc[idx_with_ms, 'motion_score'] = self.samples_df.loc[idx_with_ms].apply(self.compute_motion_score, axis=1)
      
      # flag those records that can be savely assumed as stop
      assured_stop_idx = self.samples_df[self.samples_df.motion_score == 1.0].index
      self.samples_df.loc[assured_stop_idx, 'uncertain'] = False

    # **************
    # apply stop score algorithms
    # find those records which need to be checked using the stop scores
    unsure_ids = pd.DataFrame(self.samples_df[self.samples_df.uncertain].id)

    # add the neighbouring records to be able to compute the stop scores
    # this way, not every score is computed, but only those of uncertainty
    unsure_ids['range_start'] = unsure_ids.id.apply(lambda x: x - int(self.max_window_size / 2))
    unsure_ids['range_stop'] = unsure_ids.id.apply(lambda x: x + int(self.max_window_size / 2))
    ranges = unsure_ids.apply(
      lambda row: list(range(row.range_start, row.range_stop)),
      axis=1
    ).values
    relevant_ids_flat = np.array(list(ranges)).flatten()
    relevant_ids = np.unique(relevant_ids_flat)
    run_score_algorithms_idx = self.samples_df[self.samples_df.id.isin(relevant_ids)].index
    self.samples_df['run_scores'] = False
    self.samples_df.loc[run_score_algorithms_idx, 'run_scores'] = True

    # create subset to apply algorithms on & apply in a rolling manner
    run_scores = self.samples_df[self.samples_df.run_scores].set_index('key')
    score_results = pd.DataFrame()
    score_results['id'] = run_scores.id

    # Method no. 2 - rectangle-edge-distance
    if self.settings['USE_METHOD_RDR']:
      score_results['rec_dist_score'] = run_scores.distance_to_next.rolling(
        self.settings['METHOD_RECTANGLE_DISTANCE_WINDOW_SIZE'],
        min_periods=self.settings['METHOD_RECTANGLE_DISTANCE_WINDOW_SIZE'],
        center=True
      ).apply(self.rectangle_path_distance_analysis)
    else:
      score_results['rec_dist_score'] = np.nan
    
    # Method no. 3 - bearing analysis
    if self.settings['USE_METHOD_BA']:
      score_results['bearing_score'] = run_scores.bearing.rolling(self.settings['METHOD_BEARING_ANALYSIS_WINDOW_SIZE'], min_periods=(self.settings['METHOD_BEARING_ANALYSIS_WINDOW_SIZE']-1), center=True).apply(self.bearing_analysis)
    else:
      score_results['bearing_score'] = np.nan

    # Method no. 4 - start & end analysis
    if self.settings['USE_METHOD_SEDA']:
      score_results['start_end_distance_score'] = run_scores.distance_to_next.rolling(self.settings['METHOD_START_END_DISTANCE_ANALYSIS_WINDOW_SIZE'], min_periods=self.settings['METHOD_START_END_DISTANCE_ANALYSIS_WINDOW_SIZE'], center=True).apply(self.start_end_distance_analysis)
    else:
      score_results['start_end_distance_score'] = np.nan

    # Method no. 5 - intersecting segments analysis
    if self.settings['USE_METHOD_ISA']:
      score_results = score_results.reset_index()
      run_scores = run_scores.reset_index()
      score_results['intersecting_segments_score'] = self.intersecting_segments_analysis(run_scores)
    else:
      score_results['intersecting_segments_score'] = np.nan
    
    # compute final score
    weights = [
      self.settings['METHOD_RECTANGLE_DISTANCE_RATIO_WEIGHT'],
      self.settings['METHOD_BEARING_ANALYSIS_WEIGHT'],
      self.settings['METHOD_START_END_DISTANCE_ANALYSIS_WEIGHT'],
      self.settings['METHOD_INTERSECTING_SEGMENTS_ANALYSIS_WEIGHT'],
    ]
    score_results['final_score'] = (score_results[['rec_dist_score', 'bearing_score', 'start_end_distance_score', 'intersecting_segments_score']] * weights).mean(axis=1) / np.mean(weights) 

    # filter score results
    relevant_scores = score_results[score_results.id.isin(unsure_ids.id)]

    # merge scores back to main data frame
    self.samples_df = self.samples_df.set_index('id')
    relevant_scores = relevant_scores.set_index('id')
    self.samples_df['rec_dist_score'] = relevant_scores.rec_dist_score
    self.samples_df['bearing_score'] = relevant_scores.bearing_score
    self.samples_df['start_end_distance_score'] = relevant_scores.start_end_distance_score
    self.samples_df['intersecting_segments_score'] = relevant_scores.intersecting_segments_score
    self.samples_df['score_algorithms'] = relevant_scores.final_score

    # **************
    # Method no. 6 - find stop through data gaps
    if self.settings['USE_METHOD_MDA']:
      uncertain_subset = self.samples_df[self.samples_df.uncertain].copy()
      uncertain_subset = StopGoClassifier.add_attributes(uncertain_subset, time=True, speed=True)
      stop_candidates = uncertain_subset[uncertain_subset.time_diff_to_next >= self.settings['MIN_MISSING_DATA_INTERVAL']]
      if stop_candidates.shape[0] > 0:
        self.samples_df['missing_data_stop_score'] = stop_candidates.apply(self.missing_data_analysis, axis=1)
      else:
        self.samples_df['missing_data_stop_score'] = np.nan
    else:
      self.samples_df['missing_data_stop_score'] = np.nan
    
    # **************
    # finalize; create final stop decision
    self.samples_df = self.conclude(self.samples_df)
    self.samples_df = self.samples_df.reset_index().set_index('index')
    self.samples_df = self.samples_df[['ts', 'x', 'y', 'rec_dist_score', 'bearing_score', 'start_end_distance_score', 'intersecting_segments_score', 'overall_score', 'is_stop', 'confidence']].dropna(subset=['overall_score'])


  def aggregate(self):
    self.samples_df['decision_shifted'] = self.samples_df.is_stop.shift(1)
    self.samples_df.loc[self.samples_df.iloc[0].name, 'decision_shifted'] = self.samples_df.iloc[0].is_stop
    changes = self.samples_df[self.samples_df.decision_shifted != self.samples_df.is_stop].copy()

    # distinguish singular classification results
    # no change within samples
    if changes.shape[0] == 0:
      # all samples belong to the same stop
      if self.samples_df.iloc[0].is_stop == True:
        result_stop = {
          'start': self.samples_df.iloc[0].ts,
          'stop': self.samples_df.iloc[-1].ts
        }
        final_stop_df = pd.DataFrame(result_stop, index=[0])
        self.stop_df = self.add_duration_and_average_position(final_stop_df)

      # all samples belong to the same trip; return empty data frame
      else:
        self.stop_df = pd.DataFrame(columns=['start', 'stop', 'duration', 'x', 'y'])

    # more than one stop
    else:
      def interpret_shift(row):
        if row.is_stop and not row.decision_shifted:
          return 'start'
        else:
          return 'stop'
      changes['stop_state'] = changes.apply(interpret_shift, axis=1)

      # add completing states, so each stop has a start and a stop
      change_timestamps = list(changes[['ts', 'stop_state']].values)
      if change_timestamps[0][1] == 'stop':
        change_timestamps = [np.array([self.samples_df.iloc[0].ts, 'start'])] + change_timestamps
      if change_timestamps[-1][1] == 'start':
        change_timestamps = change_timestamps + [np.array([self.samples_df.iloc[-1].ts, 'stop'])]
      changes = pd.DataFrame(change_timestamps, columns=['ts', 'stop_state'])
      changes['stop_index'] = (np.arange(0, changes.shape[0]) / 2).astype(int)

      # reshape to represent stop per row
      changes = changes.pivot(index='stop_index', columns='stop_state', values='ts')

      # add duration & average positions
      self.stop_df = self.add_duration_and_average_position(changes)
      self.samples_df = self.samples_df.drop(['decision_shifted'], axis=1)


  def filter_outliers(self):
    assert self.stop_df.shape[0] > 0, 'Filtering outliers requires at least one identified stop. Did you run `aggregate()`?'

    initial_stop_count = self.stop_df.shape[0]

    # generate scores for a sequence of stops
    stop_attributed = StopGoClassifier.add_attributes(self.stop_df, dist=True, dist_prev=True, time_to_next=True, time_to_prev=True).reset_index()
    self.overwrite_distance_with_path_length(stop_attributed)

    def stop_duration_score(row):
      # duration of stop: distinct stop > 180s, threshold 120s, min 60s
      value = row.duration
      if value == self.settings['MIN_STOP_INTERVAL']:
        value = self.settings['MIN_STOP_INTERVAL'] + 0.1
      return StopGoClassifier.compute_score(
        value,
        self.settings['MIN_STOP_INTERVAL'],
        self.settings['RELEVANT_STOP_DURATION']
      )

    def independence_score(row):
      # explains if a stop stands for itself (score > 0) or is part of a cluster score < 0)
      if stop_attributed.shape[0] == 1:
        # if no other stops are around, this stop is independent
        return 1.0

      # distinguish between first/last and middle stops
      if pd.isna(row.time_to_prev):
        time_value = row.time_to_next
      elif pd.isna(row.time_to_next):
        time_value = row.time_to_prev
      else:
        time_value = min(row.time_to_prev, row.time_to_next)
      
      if time_value > self.settings['MAX_TIME_BETWEEN_STOPS_FOR_MERGE']:
        time_score = 3.0
      else:  
        time_score = StopGoClassifier.compute_score(time_value, self.settings['MIN_TIME_BETWEEN_STOPS'], self.settings['RELEVANT_TIME_BETWEEN_STOPS'])

      if pd.isna(row.distance_to_prev):
        dist_value = row.distance_to_next
      elif pd.isna(row.distance_to_next):
        dist_value = row.distance_to_prev
      else:
        dist_value = min(row.distance_to_prev, row.distance_to_next)
      dist_score = StopGoClassifier.compute_score(dist_value, self.settings['MIN_DISTANCE_BETWEEN_STOP'], self.settings['RELEVANT_DISTANCE_BETWEEN_STOP'])
      
      if (time_score == 1.0) and (dist_score == -1):
        return -0.001 # favour distance score if the two oppose each other
      return np.mean([time_score, dist_score])

    def conclude_merge(row):
      if (row.stop_duration_score == -1.0) and (row.independence_score >= 0.0):
        return 'delete'
      elif row.independence_score < 0.0:
        return 'merge'
      else:
        return 'keep'

    def merge_direction(row):
      if row.merge_decision != 'merge':
        return np.nan

      # indicates if the closest neighbour is the previous one (-1) or the next one (1)
      if (row.next_merge_decision != 'delete'):
        if pd.isna(row.distance_to_prev) or ((min(row.distance_to_prev, row.distance_to_next) == row.distance_to_next) and (row.time_to_next <= self.settings['MAX_TIME_BETWEEN_STOPS_FOR_MERGE'])):
          return 1
      
      # ensure merging backwards is allowed
      if (row.prev_merge_decision != 'delete'):
        if not pd.isna(row.distance_to_prev) and ((min(row.distance_to_prev, row.distance_to_next) == row.distance_to_prev) and (row.time_to_prev <= self.settings['MAX_TIME_BETWEEN_STOPS_FOR_MERGE'])):
          return -1
      
      # if we can't merge backwards, nor forwards, delete this merge candidate
      return np.nan

    # add merge metrics to data frame
    stop_attributed['stop_duration_score'] = stop_attributed.apply(stop_duration_score, axis=1)
    stop_attributed['independence_score'] = stop_attributed.apply(independence_score, axis=1)
    stop_attributed['merge_decision'] = stop_attributed.apply(conclude_merge, axis=1)
    stop_attributed['next_merge_decision'] = stop_attributed.merge_decision.shift(-1)
    stop_attributed['prev_merge_decision'] = stop_attributed.merge_decision.shift(1)
    stop_attributed['merge_direction'] = stop_attributed.apply(merge_direction, axis=1)
    if self.debug_stop_merge_df is None:
      self.debug_stop_merge_df = stop_attributed.copy()

    # remove irrelevant stops
    stop_attributed = stop_attributed[(stop_attributed.merge_decision != 'delete') & ~((stop_attributed.merge_decision == 'merge') & pd.isna(stop_attributed.merge_direction))].copy()

    # stop here if there is no meaningful stop remaining after the deletion
    if (stop_attributed.shape[0] == 0) or ((stop_attributed.shape[0] == 1) and (stop_attributed.iloc[0].stop_duration_score == -1)):
      final_stop_df = StopGoClassifier.empty_stops_df()

    else:
      # if only one merger was identified, continue with this one
      if stop_attributed.shape[0] == 1:
        final_stop_df = stop_attributed[['start', 'stop']].copy()

      else:
        # merge stops by adding to closest stop
        stop_attributed['id'] = range(stop_attributed.shape[0])
        stop_attributed = stop_attributed.set_index('id')
        stop_attributed['merged'] = False
        merger_ids = stop_attributed[stop_attributed.merge_decision == 'merge'].index.values

        # merge stops with the closest neigbours
        for index in merger_ids:
          merger = stop_attributed.loc[index]

          # select stop to merge with
          neighbour_index = index + merger.merge_direction
          closest_neighbour = stop_attributed.loc[neighbour_index]

          # if the closest option was merged already, the current stop was modified and sould not be processed further
          if closest_neighbour.merged == True:
            continue

          # merge timestamps & overwrite timestamps of the keeper entry
          stop_attributed.loc[neighbour_index, 'start'] = min(closest_neighbour.start, merger.start)
          stop_attributed.loc[neighbour_index, 'stop'] = max(closest_neighbour.stop, merger.stop)
          stop_attributed.loc[index, 'merged'] = True # no other stop should merge with this one

        remaining_stops = stop_attributed[stop_attributed.merged == False]
        final_stop_df = remaining_stops[['start', 'stop']].copy()

    final_stop_df.reset_index(drop=True, inplace=True)
    final_stop_df.columns.name = 'id'

    # add duration & average positions
    self.stop_df = self.add_duration_and_average_position(final_stop_df)

    # decide if a second filter & merge iteration should be done
    # if a merger is between two keepers, a second iteration will check if
    # the two resulting keepers should be merged
    if (initial_stop_count != self.stop_df.shape[0]) and (self.stop_df.shape[0] > 0):
      self.filter_outliers()

  def isolate_trip_samples(self):
    if len(self.stop_df) == 0:
      self.trip_df = pd.DataFrame([[self.samples_df.ts.iloc[0], self.samples_df.ts.iloc[-1]]], columns=['start', 'stop'])
      self.trip_samples_df = self.samples_df
      return

    trips = pd.DataFrame(columns=['start', 'stop'])

    # isolate trip points
    if len(self.stop_df) >= 2:
        trips = self.stop_df[['start', 'stop']].copy()
        trips.start = trips.stop
        trips.stop = self.stop_df.start.shift(-1)
        trips = trips.iloc[:-1]

    # check if there are samples before & after the first & last recognized stop
    if self.samples_df.iloc[0].ts < self.stop_df.iloc[0].start:
        trips.loc[len(trips)] = {'start': self.samples_df.iloc[0].ts, 'stop': self.stop_df.iloc[0].start}
    if self.samples_df.iloc[-1].ts > self.stop_df.iloc[-1].stop:
        trips.loc[len(trips)] = {'start': self.stop_df.iloc[-1].stop, 'stop': self.samples_df.iloc[-1].ts}

    # define trip list
    self.trip_df = trips.sort_values('start').reset_index()[['start', 'stop']]
    if len(self.trip_df) > 0:
      self.trip_df['duration'] = trips.apply(lambda row: (row.stop - row.start).total_seconds(), axis=1)
    else:
      self.trip_df = pd.DataFrame(columns=['start', 'stop', 'duration'])

    # isolate trip related GPS samples
    self.trip_samples_df = pd.DataFrame()
    for _, trip in trips.iterrows():
        self.trip_samples_df = pd.concat([
          self.trip_samples_df,
          self.samples_df[(self.samples_df.ts >= trip.start) & (self.samples_df.ts <= trip.stop)]
        ])



  ##################################################
  # stop score algorithms
  ##################################################
  def rectangle_path_distance_analysis(self, window):
    xs = np.array(list(map(lambda x: x[0], window.index.values)))
    ys = np.array(list(map(lambda x: x[1], window.index.values)))

    # identify furthest points
    points = np.array(list(window.index.values))  

    try:
      convexhull = spatial.ConvexHull(points)
      convhull_vertices = points[convexhull.vertices]
      dist_mat = spatial.distance_matrix(convhull_vertices, convhull_vertices)
      idx_p1, idx_p2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
      x1, y1 = convhull_vertices[idx_p1]
      x2, y2 = convhull_vertices[idx_p2]

    except spatial.QhullError:
      # assume a collapsing QHull; infer max distance from first & last point in list
      x1, y1 = points[0]
      x2, y2 = points[-1]
    
    max_distance_between_points = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    # path distance
    cummulative_distance = window.values.sum()

    # ratio
    rectangle_distance_ratio = cummulative_distance / max_distance_between_points

    return StopGoClassifier.compute_score(
      rectangle_distance_ratio,
      1.0,
      self.settings['METHOD_RECTANGLE_DISTANCE_RATIO_UPPER_CUTOFF'],
      self.settings['METHOD_RECTANGLE_DISTANCE_RATIO_THRESHOLD']
    )

  def bearing_analysis(self, window):
    sorted_slice = np.sort(window)[1:-1]
    mean_abs_angle_diff = np.mean(sorted_slice)

    return StopGoClassifier.compute_score(
      mean_abs_angle_diff,
      self.settings['METHOD_BEARING_ANALYSIS_LOWER_CUTOFF'],
      self.settings['METHOD_BEARING_ANALYSIS_UPPER_CUTOFF'],
      self.settings['METHOD_BEARING_ANALYSIS_THRESHOLD']
    )

  def start_end_distance_analysis(self, window):
    xs = np.array(list(map(lambda x: x[0], window.index.values)))
    ys = np.array(list(map(lambda x: x[1], window.index.values)))

    # define mean start & end
    start_x = xs[:2].mean()
    start_y = ys[:2].mean()
    end_x = xs[-2:].mean()
    end_y = ys[-2:].mean()

    # compute distance
    distance = np.sqrt(np.square(start_x - end_x) + np.square(start_y - end_y))
    
    # transform to score in [-1, 1]
    score = StopGoClassifier.compute_score(
      distance,
      self.settings['METHOD_START_END_DISTANCE_ANALYSIS_LOWER_CUTOFF'],
      self.settings['METHOD_START_END_DISTANCE_ANALYSIS_UPPER_CUTOFF'],
      self.settings['METHOD_START_END_DISTANCE_ANALYSIS_THRESHOLD']
    )
    return score * -1

  def intersecting_segments_analysis(self, samples_df):
    # create sparse comparison matrix
    segments = np.array([
      samples_df.x.values[:-1],
      samples_df.y.values[:-1],
      samples_df.x.values[1:],
      samples_df.y.values[1:],
    ])

    row_count = len(samples_df) - 1
    intersection_matrix = np.zeros((row_count, row_count))
    for row_idx in range(row_count):
      for col_idx in range(row_idx + 2, min(row_idx + self.settings['METHOD_INTERSECTING_SEGMENTS_ANALYSIS_WINDOW_SIZE'], row_count)):
        ax = segments[0, row_idx]
        ay = segments[1, row_idx]
        bx = segments[2, row_idx]
        by = segments[3, row_idx]
        cx = segments[0, col_idx]
        cy = segments[1, col_idx]
        dx = segments[2, col_idx]
        dy = segments[3, col_idx]
        intersection_matrix[row_idx, col_idx] = StopGoClassifier.intersect(ax, ay, bx, by, cx, cy, dx, dy)

    # count & score windowed intersections
    def count_intersections(window):
      a, b = window.index[0], window.index[-1]
      intersections = intersection_matrix[a:b, a:b].sum()
      return StopGoClassifier.compute_score(
        intersections,
        0,
        self.settings['METHOD_INTERSECTING_SEGMENTS_ANALYSIS_UPPER_CUTOFF'],
        self.settings['METHOD_INTERSECTING_SEGMENTS_ANALYSIS_THRESHOLD']
      )

    # iterate the dataframe in a rolling manner
    scores = samples_df.x.rolling(
      self.settings['METHOD_INTERSECTING_SEGMENTS_ANALYSIS_WINDOW_SIZE'],
      min_periods=self.settings['METHOD_INTERSECTING_SEGMENTS_ANALYSIS_WINDOW_SIZE'],
      center=True
    ).apply(count_intersections)
    return scores

  def missing_data_analysis(self, row):
    speed = row.speed * 3.6 # km/h

    score = StopGoClassifier.compute_score(
      speed,
      self.settings['METHOD_MISSING_DATA_ANALYSIS_LOWER_CUTOFF'],
      self.settings['METHOD_MISSING_DATA_ANALYSIS_UPPER_CUTOFF'],
      self.settings['METHOD_MISSING_DATA_ANALYSIS_THRESHOLD']
    )
    return score * -1



  ##################################################
  # helper
  ##################################################
  def compute_motion_score(self, row):
    limited = min(self.settings['MOTION_SCORE_UPPER_CUTOFF'], max(self.settings['MOTION_SCORE_LOWER_CUTOFF'], row.motion_score))
    shifted = limited - self.settings['MOTION_SCORE_THRESHOLD']

    if shifted < 0:
      return (1 / (self.settings['MOTION_SCORE_THRESHOLD'] - self.settings['MOTION_SCORE_LOWER_CUTOFF'])) * shifted * -1
    else:
      return (1 / (self.settings['MOTION_SCORE_UPPER_CUTOFF'] - self.settings['MOTION_SCORE_THRESHOLD'])) * shifted * -1

  def find_stop_center_from_timestamps(self, row):
    # row = changes.iloc[0]
    if pd.isna(row.start):
      relevant_samples = self.samples_df[self.samples_df.ts <= row.stop]
    elif pd.isna(row.stop):
      relevant_samples = self.samples_df[self.samples_df.ts >= row.start]
    else:
      relevant_samples = self.samples_df[(self.samples_df.ts >= row.start) & (self.samples_df.ts <= row.stop)]

    x = relevant_samples.x.median()
    y = relevant_samples.y.median()
    return (x, y)
  
  def add_duration_and_average_position(self, df):
    if df.shape[0] > 0:
      df['duration'] = df.apply(lambda row: (row.stop - row.start).total_seconds(), axis=1)
      stop_centers = df.apply(self.find_stop_center_from_timestamps, axis=1)
      df['x'] = stop_centers.apply(lambda x: x[0])
      df['y'] = stop_centers.apply(lambda x: x[1])
    return df

  @staticmethod
  def bearing_deviation(ax, ay, bx, by, cx, cy):
    # https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd

    if np.isnan(ax) or np.isnan(cx):
      return np.nan

    a = np.array([ax, ay])
    b = np.array([bx, by])
    c = np.array([cx, cy])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    if cosine_angle < -1:
      cosine_angle = -1
    elif cosine_angle > 1:
      cosine_angle = 1

    difference_deg = np.rad2deg(np.arccos(cosine_angle) - np.pi)
    return np.abs(difference_deg)

  @staticmethod
  def add_attributes(df, dist=False, dist_prev=False, time=False, speed=False, bearing=False, time_to_next=False, time_to_prev=False):
    if speed or dist or bearing:
      df['next_x'] = df.x.shift(-1)
      df['next_y'] = df.y.shift(-1)
      df['distance_to_next'] = np.sqrt(np.square(df.x - df.next_x) + np.square(df.y - df.next_y))

    if dist_prev or bearing:
      df['prev_x'] = df.x.shift(1)
      df['prev_y'] = df.y.shift(1)
      df['distance_to_prev'] = np.sqrt(np.square(df.x - df.prev_x) + np.square(df.y - df.prev_y))

    if speed or time:
      df['next_ts'] = df.ts.shift(-1)
      df['time_diff_to_next'] = df.apply(lambda row: (row.next_ts - row.ts).total_seconds(), axis=1)

    if speed:
      def compute_speed(row):
        if row.time_diff_to_next > 0:
          return row.distance_to_next / row.time_diff_to_next
        else:
          return np.nan
      df['speed'] = df.apply(compute_speed, axis=1)

    if bearing:
      df['bearing'] = df.apply(lambda row: StopGoClassifier.bearing_deviation(row.prev_x, row.prev_y, row.x, row.y, row.next_x, row.next_y), axis=1)

    if time_to_next:
      df['next_start'] = df.start.shift(-1)
      df['time_to_next'] = df.apply(lambda row: (row.next_start - row.stop).total_seconds(), axis=1)

    if time_to_prev:
      df['prev_stop'] = df.stop.shift(1)
      df['time_to_prev'] = df.apply(lambda row: (row.start - row.prev_stop).total_seconds(), axis=1)

    return df
  
  @staticmethod
  def path_length_between(df, start, stop):
    samples_between = df[(df.ts >= start) & (df.ts <= stop)].copy()
    if len(samples_between) < 2:
      return np.nan

    StopGoClassifier.add_attributes(samples_between, dist=True)
    return samples_between.distance_to_next.sum()

  def overwrite_distance_with_path_length(self, df):
    # overwrite distance to next/prev stop with the sum of distances between samples
    df['next_start'] = df.start.shift(-1)
    df['path_length_to_next'] = df.apply(lambda row: StopGoClassifier.path_length_between(self.samples_df, row.stop, row.next_start), axis=1)
    df['path_length_to_prev'] = df.path_length_to_next.shift(1)

    # use the maximum of the euclidean distance and the path distance
    # if there are records missing, the path distance could be less than the eu. dist.
    df['distance_to_next'] = df.apply(lambda row: max(row.distance_to_next, row.path_length_to_next), axis=1)
    df['distance_to_prev'] = df.apply(lambda row: max(row.distance_to_prev, row.path_length_to_prev), axis=1)
  
  @staticmethod
  def intersect(ax, ay, bx, by, cx, cy, dx, dy):
    # http://www.bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)
  
  @staticmethod
  def compute_score(value, lower_cutoff, upper_cutoff, threshold=None):
    if pd.isna(value):
      return np.nan
    if threshold is None:
      threshold = np.mean([lower_cutoff, upper_cutoff])

    shifted = min(upper_cutoff, max(lower_cutoff, value)) - threshold
    if shifted < 0:
      return (1 / (threshold - lower_cutoff)) * shifted
    return (1 / (upper_cutoff - threshold)) * shifted

  def conclude(self, df):
    def select_score(row):
      # row = relevant_slice.iloc[50]
      # uncertain records are those with at least somewhat ACC trip
      if row.uncertain:
        if not self.settings['USE_METHOD_MDA'] or pd.isna(row.missing_data_stop_score):
          # scores are the combination from all methodic Methodes
          return row.score_algorithms
        else:
          # missing data stop is computed by analysing the time and distance of two consecutive records
          return row.missing_data_stop_score

      # if a record is 'certain', the ACC analysis indicates no physical trip
      return row.motion_score

    def overwrite_rolling_average(row):
      # overwrite if the missing data analysis concludes stop
      if row.missing_data_stop_score > 0:
        return row.missing_data_stop_score
      elif pd.isna(row.overall_score):
        return row.score_algorithms
      return row.overall_score

    df['overall_score'] = df.apply(select_score, axis=1)
    smoothing_window_size = 5
    df.overall_score = df.overall_score.rolling(smoothing_window_size, min_periods=smoothing_window_size, center=True).mean()
    df.overall_score = df.apply(overwrite_rolling_average, axis=1)

    df['is_stop'] = df.overall_score > 0
    df['confidence'] = df.overall_score.abs()
    return df
  
  @staticmethod
  def empty_stops_df():
    return pd.DataFrame(columns=['start', 'stop', 'duration', 'x', 'y'])