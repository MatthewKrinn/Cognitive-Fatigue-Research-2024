# =========================
# Imports / Dependencies
# =========================


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import os
import shutil
import glob
from pathlib import Path


# =========================
# Path Macros
# =========================


REPO_DIRECTORY = Path.cwd()

DATA_DIRECTORY = os.path.join(REPO_DIRECTORY, "data")

FATIGUESET_DIRECTORY = os.path.join(DATA_DIRECTORY, "fatigueset")

# This allows the different ntoebooks to read / write to different folders using the same macro
class Config:
    FATIGUESET_AGG_DIRECTORY = None


# =========================
# Data Creation and Moving
# =========================


def move_data(transfer_files):
    """for each person in fatigueset folder, copy the csv files in transfer_files from fatigueset/person/low, medium, high to fatigueset_agg_data/person_metadata and create high med low folders"""

    for person in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        for fatigue_level in ['low', 'medium', 'high']:
            for file in transfer_files:
                src = os.path.join(FATIGUESET_DIRECTORY, person, fatigue_level, file)
                dst = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f'{person}_metadata', fatigue_level, file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
        print(f'Generated {person}_metadata in {Config.FATIGUESET_AGG_DIRECTORY}')

def file_combine(filenames, mapping, agg_time, extension):
    """Return a combined csv of all files in filenames with the columns renamed according to mapping, for each person.
    filenames: list of filenames to combine for each person
    mapping: dictionary of column name mappings. Key --> Value : filename --> column prefix
        EX: {'exp_crt.csv': 'crt', 'exp_nback.csv': 'nback'}
    agg_time: time to aggregate the data
    extension: file extension of the files in filenames. DO NOT INCLUDE . IN EXTENSION

    Returns: combined_csv for each person
    """
    
    # want contains the mapping keys (names of selected time series data) + extension
    want = [f'{key}.{extension}' for key in mapping.keys()]

    csvs = []
    
    for f in filenames:
        if f not in want:
            continue

        #DEBUG: Uncomment to see path
        # print(f'{os.getcwd()=}')

        file = pd.read_csv(f)

        
        file['DateTime'] = file['timestamp'].apply(timestamp_to_UTC)


        # Convert the DateTime column to a datetime object, set as index, and drop as a column
        file['DateTime'] = pd.to_datetime(file['DateTime'])
        file.set_index('DateTime', inplace=True)
        file.drop(columns=['timestamp'], inplace=True)


        # add prefix to every column based on mapping variable and csv name
        for key in mapping:
            if key in f:
                file.columns = [f'{mapping[key]}_{col}' for col in file.columns]
        

        # Aggregate information together inside 1s intervals
        file = file.resample(agg_time).mean()

        csvs.append(file)

    combined_csv = pd.concat(csvs, axis=1)


    #DEBUG: Uncomment to see shape of csv, make sure it's correct
    # print(f'combined shape of csv: {combined_csv.shape}')

    # Sometimes pandas adds this random column, delete it if it exists
    unnamed_columns = combined_csv.filter(like='Unnamed:').columns
    combined_csv.drop(unnamed_columns, axis=1, inplace=True)

    return combined_csv

def spawn_data(mapping, agg_time, extension):
    """Spawns personalized person.csv and metadata/ folder"""

    os.chdir(FATIGUESET_DIRECTORY)
    
    all_folders = [name for name in os.listdir(".") if os.path.isdir(name)]
    for folder in all_folders:

        if folder in ['metadata.csv', 'pre_task_survey.xlsx', 'preliminary_questionnaire.xlsx', 'README.md']:
            continue

        person_csv = pd.DataFrame()

        #DEBUG: Uncomment to identify working folder
        # print(f'folder: {folder}')
        
        os.chdir(str(folder))

        for inner in ['high', 'medium', 'low']:

            os.chdir(str(inner))

            #iterate through all files in folder
            filenames_in_folder = [i for i in glob.glob('*.{}'.format(extension))]
            
            #DEBUG: Uncomment to list all files in working folder
            # print(f'all filenames: {all_filenames}')

            csv = file_combine(filenames_in_folder, mapping, agg_time, extension)

            #add a column for the folder name
            csv['Session'] = inner

            person_csv = pd.concat([person_csv, csv], axis=0)

            #go back to the outside directory
            os.chdir("..")
        
        os.chdir(Config.FATIGUESET_AGG_DIRECTORY)

        person_csv.to_csv(f"{folder}_data.csv", index=True, encoding='utf-8-sig')
        os.chdir("..")
        os.chdir(FATIGUESET_DIRECTORY)
        print(f'Generated {folder}_data.csv in {Config.FATIGUESET_AGG_DIRECTORY}')
    os.chdir(REPO_DIRECTORY)


# =================================
# Preprocessing and Time Functions
# =================================


def fill_utc_gaps(markers):
    """ Helper function for fill_marker_nans().
    In a person / session's markers csv, some times associated with an event are empty. 
    get_time_for_event() throws an error if it reads any NaNs. 
    To keep the data monotonic, fill these gaps with the previous time + 10 seconds.
    Note: No important events have null values, so this is ONLY to make get_time_for_event() not throw errors.
    """

    df = markers.copy()

    nan_mask = df['utcTime'].isna()

    df.loc[nan_mask, 'utcTime'] = df['utcTime'].ffill() + 10000

    return df

def fill_marker_nans():
    """Fill NaNs in all exp_markers.csv files with the time of previous event + 10 seconds"""
    for person in range(1, 13):
        for session in ['low', 'medium', 'high']:
            person, session = remap_person_and_session(person, session)
            path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_metadata", session, "exp_markers.csv")
            markers = pd.read_csv(path)
            markers = fill_utc_gaps(markers)
            markers.to_csv(path, index=False)


def timestamp_to_UTC(timestamp):
    """Simple function that converts Unix Timestamp (type of timestamp in fatigeset csvs) to DateTime object"""
    # Convert milliseconds to seconds
    timestamp_in_seconds = timestamp / 1000

    # Convert timestamp to datetime object
    datetime_object = datetime.utcfromtimestamp(timestamp_in_seconds)

    # Format the datetime object as a string
    formatted_datetime = datetime_object.strftime('%Y-%m-%d %H:%M:%S.%f')

    return formatted_datetime

def remap_person_and_session(person, session):
    """Remap ints and improperly formatted strings. Should be called before any other function uses person or session identifiers."""
    if len(str(person)) == 1:
        person = '0' + str(person)
    else:
        person = str(person)
    # Depreciated
    # if len(str(session)) == 1:
    #     session = '0' + str(session)

    return person, session

def make_specific_events_unique(file_path):
    """Helper function.
    Example: There are 3 submit_survey events, they will be renamed to submit_survey_1, submit_survey_2, submit_survey_3.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # List of events that we know will repeat
    repeat_events = ['submit_survey', 'start_crt', 'start_nback']
    
    # Counter for each repeating event
    event_counters = {event: 0 for event in repeat_events}
    
    # Function to modify specific event markers
    def modify_specific_events(event):
        if event in repeat_events:
            event_counters[event] += 1
            return f"{event}_{event_counters[event]}"
        return event
    
    # Apply the modification to the eventMarker column
    df['eventMarker'] = df['eventMarker'].apply(modify_specific_events)
    
    # Save the modified DataFrame back to CSV
    df.to_csv(file_path, index=False)
    
    print(f"File {file_path} has been updated with unique event markers for specific events.")

def add_numbers_to_marker_csv():
    """Preprocessing step that renames duplicated eventMarkers from subsequent measurement periods."""
    for person in range(1, 13):
        for session in ['low', 'medium', 'high']:
            person, session = remap_person_and_session(person, session)
            path = os.path.join(FATIGUESET_DIRECTORY, person, str(session), "exp_markers.csv")
            make_specific_events_unique(path)

def change_session_nums_to_session_identifiers():
    """Renames each person's session folders from integers to the activity type.
    fatigueset/metadata.csv contains mapping of a person's integer sessions to the activity type.
    """

    base_dir = FATIGUESET_DIRECTORY

    # Load metadata.csv into a DataFrame
    metadata_file = os.path.join(FATIGUESET_DIRECTORY, 'metadata.csv')

    metadata_df = pd.read_csv(metadata_file, dtype=str)

    for index, row in metadata_df.iterrows():
        # print(row['participant_id'])
        # print(row['low_session'])
        # print(row['medium_session'])
        # print(row['high_session'])
        # print('----------------------')

        person = row['participant_id']

        low = row['low_session']
        medium = row['medium_session']
        high = row['high_session']

        low_path = os.path.join(base_dir, person, low)
        # rename this folder to low
        new_low_path = os.path.join(base_dir, person, 'low')
        os.rename(low_path, new_low_path)
        # print(f"Renamed folder '{person}/{low}' to '{person}/low'")

        # do the same for medium and high
        medium_path = os.path.join(base_dir, person, medium)
        new_medium_path = os.path.join(base_dir, person, 'medium')
        os.rename(medium_path, new_medium_path)
        # print(f"Renamed folder '{person}/{medium}' to '{person}/medium'")

        high_path = os.path.join(base_dir, person, high)
        new_high_path = os.path.join(base_dir, person, 'high')
        os.rename(high_path, new_high_path)
        # print(f"Renamed folder '{person}/{high}' to '{person}/high'")

        print(f"Renamed session folders for person {person}")

def get_time_of_event(event, person, session):
    """Given an event, person, and session, return the time of the event by reading relevent exp_markers.csv"""
    person, session = remap_person_and_session(person, session)
    
    # path_to_markers = rf"C:/Users/639766/Documents/Internship/fatigueset_agg_data/{person}_metadata/{session}/exp_markers.csv"
    path_to_markers = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_metadata", session, "exp_markers.csv")
    
    markers = pd.read_csv(path_to_markers)

    markers['utcTime'] = markers['utcTime'].apply(timestamp_to_UTC)
    markers['utcTime'] = pd.to_datetime(markers['utcTime'])

    x = markers[markers['eventMarker'] == event]['utcTime'].values[0]

    return x

def where_is_time(utc_time, person, session):
    """Given input of utc time, person, session, return the marker that it is closest to"""

    person, session = remap_person_and_session(person, session)
    # Load the metadata

    path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_metadata", session, "exp_markers.csv")

    metadata = pd.read_csv(path)
    # Find the start and end of the baseline
    metadata['utcTime'] = metadata['utcTime'].apply(timestamp_to_UTC)
    metadata['utcTime'] = pd.to_datetime(metadata['utcTime'])

    # Find the closest marker, but the MARKER HAS TO BE BEFORE THE TIME

    utc_time = pd.to_datetime(timestamp_to_UTC(utc_time))
    closest_marker = metadata.loc[metadata['utcTime'] <= utc_time, 'eventMarker'].iloc[-1]

    return closest_marker

def initialize_aggregated_data(transfer_files, mapping):
    """Does all preprocessing steps to generate the aggregated data directly after copying fatigueset folder into data/"""

    # Only go through the renaming procecss if necessary
    if os.path.isdir(
        os.path.join(FATIGUESET_DIRECTORY, '01', '01')
    ):
        change_session_nums_to_session_identifiers()
    else:
        print('Session folder has already been renamed')

    
    if pd.read_csv(os.path.join(FATIGUESET_DIRECTORY, '01', 'low', 'exp_markers.csv')).iloc[3].values[1] != 'submit_survey_1':
        add_numbers_to_marker_csv()
    else:
        print('Markers have already been renamed')
    


    move_data(transfer_files)

    extension = 'csv'


    aggregation_time = '1s'

    spawn_data(mapping, aggregation_time, extension)

    fill_marker_nans()
    


# =======================================
# Plotting Functions for groupby / apply
# =======================================


def plot(x, feature):
    # increase size of plots

    plt.plot(x[feature])

    person = x.iloc[0]['person']
    session = x.iloc[0]['Session']


    # plot vertical lines at start and end of fatigue

    start_fatigue = get_time_of_event('start_fatigue', person, session)
    end_fatigue = get_time_of_event('end_fatigue', person, session)
    plt.axvline(start_fatigue, color='r', linestyle='--')
    plt.axvline(end_fatigue, color='r', linestyle='--')

    start_baseline = get_time_of_event('start_baseline', person, session)
    end_baseline = get_time_of_event('end_baseline', person, session)
    plt.axvline(start_baseline, color='g', linestyle='--')
    plt.axvline(end_baseline, color='g', linestyle='--')

    start_activity = get_time_of_event('start_activity', person, session)
    end_activity = get_time_of_event('end_activity', person, session)
    plt.axvline(start_activity, color='b', linestyle='--')
    plt.axvline(end_activity, color='b', linestyle='--')

    

    plt.title(f"Person {person} Session {session}")
    plt.show()



def sub(x, feature):
    """Uses plot as a helper function to display each person's data in a single row.
    Green: Baseline, Blue: Physical Activity, Red: Fatiguing Activity."""

    person = x.iloc[0]['person']
    #subplot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for i, session in enumerate(['low', 'medium', 'high']):
        ax[i].tick_params(axis='x', rotation=45)
        ax[i].plot(x[x['Session'] == session][feature])    

        # plot vertical lines at start and end of events
        start_fatigue = get_time_of_event('start_fatigue', person, session)
        end_fatigue = get_time_of_event('end_fatigue', person, session)
        ax[i].axvline(start_fatigue, color='r', linestyle='--')
        ax[i].axvline(end_fatigue, color='r', linestyle='--')

        start_baseline = get_time_of_event('start_baseline', person, session)
        end_baseline = get_time_of_event('end_baseline', person, session)
        ax[i].axvline(start_baseline, color='g', linestyle='--')
        ax[i].axvline(end_baseline, color='g', linestyle='--')

        start_activity = get_time_of_event('start_activity', person, session)
        end_activity = get_time_of_event('end_activity', person, session)
        ax[i].axvline(start_activity, color='y', linestyle='--')
        ax[i].axvline(end_activity, color='y', linestyle='--')

        scores = get_fatigue_scores_df(person, session)

        baseline = scores[scores['measurementNumber'] == 0]['mentalFatigueScore'].item()
        activity = scores[scores['measurementNumber'] == 1]['mentalFatigueScore'].item()
        fatigue = scores[scores['measurementNumber'] == 2]['mentalFatigueScore'].item()

        ax[i].set_title(f'{session}: {baseline:.2f} --> {activity:.2f} --> {fatigue:.2f}')
        ax[i].set_ylabel('Chest HRV')
        ax[i].set_xlabel('Timestamp')


    fig.suptitle(f"Person {person}")

    plt.show()


# ================
# Data Generation 
# ================

def get_fatigue_scores_df(person, session):
    """Helper Function. Obtains the fatigue scores for a person and session."""
    person, session = remap_person_and_session(person, session)
    path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_metadata", session, "exp_fatigue.csv")
    df = pd.read_csv(path)
    return df

def get_stacked_fatigue_scores():
    """Depreciated"""
    df = None
    for person in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        for session in ['low', 'medium', 'high']:
            person, session = remap_person_and_session(person, session)
            x = get_fatigue_scores_df(person, session)
            x['person'] = person
            x['Session'] = session
            try:
                df = pd.concat([df, x])
            except:
                df = x
    
    return df

def get_crt_metrics(person, session):
    """Helper Function. Obtains the CRT metrics for a person and session."""
    person, session = remap_person_and_session(person, session)

    path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_metadata", session, "exp_crt.csv")

    crt_df = pd.read_csv(path)

    # Group by measurementBlock and then find isCorrectResponse
    accuracy_by_block = crt_df.groupby('measurementBlock')['isCorrectResponse'].sum() / 36 # always 36 trials per block

    mean_response_time_by_block = crt_df.groupby('measurementBlock')['responseTime'].mean()

    return accuracy_by_block, mean_response_time_by_block

def get_nback_metrics(person, session):
    """Helper Function. Obtains the N-Back metrics for a person and session."""
    person, session = remap_person_and_session(person, session)

    path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_metadata", session, "exp_nback.csv")

    nback_df = pd.read_csv(path)

    # Group by measurementBlock and then find isCorrectResponse
    accuracy_by_block = nback_df.groupby('measurementBlock')['isCorrectResponse'].sum() / 20 # always 20 trials per block

    mean_response_time_by_block = nback_df.groupby('measurementBlock')['responseTime'].mean()

    return accuracy_by_block, mean_response_time_by_block

def get_person_metrics(person):
    """Helper Function. Obtains the metrics for a person."""
    df = None
    for session in ['low', 'medium', 'high']:
        accuracy_by_block, mean_response_time_by_block = get_crt_metrics(person, session)
        x = pd.DataFrame(accuracy_by_block)
        x['mean_response_time'] = mean_response_time_by_block
        x['person'] = person
        x['session'] = session
        # rename isCorrectResponse column to crt_accuracy
        x.reset_index(inplace=True)
        x.rename(columns={'isCorrectResponse': 'crt_accuracy', 'measurementBlock' : 'period', 'mean_response_time' : 'crt_mean_response_time'}, inplace=True)
        x['period'] = x['period'].map({
            0: 'measurement1',
            1: 'measurement2',
            2: 'measurement3'
        })


        accuracy_by_block, mean_response_time_by_block = get_nback_metrics(person, session)
        x['nback_mean_response_time'] = mean_response_time_by_block
        x['nback_accuracy'] = accuracy_by_block

        # rename columns so it goes person, session, block and then the other ones
        cols_to_move = ['person', 'session', 'period']
        remaining_columns = [col for col in x.columns if col not in cols_to_move]
        x = x[cols_to_move + remaining_columns]

        # add fatigue scores from exp_fatigue.csv
        fatigue_scores = get_fatigue_scores_df(person, session)
        fatigue_scores.reset_index(inplace=True, drop=True)
        # fatigue_scores.rename(columns={'measurementBlock': 'block'}, inplace=True)
        fatigue_scores = fatigue_scores[['mentalFatigueScore', 'physicalFatigueScore']]
        x = pd.concat([x, fatigue_scores], axis=1)

        try:
            df = pd.concat([df, x])
        except:
            df = x
    
    return df

def get_stacked_person_csvs():
    df = None
    for person in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:

        path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_data.csv")

        x = pd.read_csv(path, parse_dates=['DateTime'])
        x['person'] = person
        try:
            df = pd.concat([df, x])
        except:
            df = x


    df.set_index('DateTime', inplace=True)

    columns_to_move = ['person', 'Session']  # specify in the order you want them moved
    new_order = columns_to_move + [col for col in df.columns if col not in columns_to_move]
    df = df[new_order]

    return df

def stack_people_metrics():
    """Helper Function. Stacks all person metrics into a single DataFrame with person column identifiers."""
    df = None
    for person in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        x = get_person_metrics(person)
        try:
            df = pd.concat([df, x])
        except:
            df = x
    df.reset_index(inplace=True, drop=True)
    return df



def add_period_column(group):
    """Adds period column to df grouped by person and session."""
    person = group['person'].iloc[0]
    session = group['Session'].iloc[0]

    time0 = get_time_of_event('start_baseline', person, session)

    time1 = get_time_of_event('end_baseline', person, session)

    time2 = get_time_of_event('start_activity', person, session)

    time3 = get_time_of_event('end_activity', person, session)

    time4 = get_time_of_event('start_fatigue', person, session)

    time5 = get_time_of_event('end_fatigue', person, session)

    time6 = get_time_of_event('end_session', person, session)

    group.loc[(group.index >= time0) & (group.index < time1), 'period'] = 'baseline'

    group.loc[(group.index >= time1) & (group.index < time2), 'period'] = 'measurement1'

    group.loc[(group.index >= time2) & (group.index < time3), 'period'] = 'physical_activity'

    group.loc[(group.index >= time3) & (group.index < time4), 'period'] = 'measurement2'

    group.loc[(group.index >= time4) & (group.index < time5), 'period'] = 'mental_activity'

    group.loc[(group.index >= time5) & (group.index < time6), 'period'] = 'measurement3'

    return group

def interpolate_df(df):
    """Helper Function. Given a DataFrame, interpolate the chest_hrv column."""
    def inter(x):
        x['chest_hrv'] = x['chest_hrv'].interpolate(method='linear')
        return x
    interpolated_df = df.groupby(['person', 'Session'], group_keys = False)[df.columns].apply(inter)

    return interpolated_df

def fill_extras(df):
    """Helper Function. Given a DataFrame, fill the NaNs at beginning and end with ffill and bfill."""
    def fffill(x):
        x['chest_hrv'] = x['chest_hrv'].ffill()
        return x
    def bffill(x):
        x['chest_hrv'] = x['chest_hrv'].bfill()
        return x
    interpolated_df = df.groupby(['person', 'Session'], group_keys = False)[df.columns].apply(fffill)
    interpolated_df = df.groupby(['person', 'Session'], group_keys = False)[df.columns].apply(bffill)

    return interpolated_df

def completely_interpolate(df):
    """Given a DataFrame, interpolate the chest_hrv column, fill the NaNs at beginning and end with ffill and bfill."""
    df = interpolate_df(df)
    df = fill_extras(df)

    return df

def attach_period_column(interpolated_df):
    """Add column to time series df that specifies the period of the measurement for each time index"""
    interpolated_df = interpolated_df.groupby(['person', 'Session'], group_keys = False)[interpolated_df.columns].apply(add_period_column)

    # Reorder columns
    moving_cols = ['person', 'Session', 'period']
    rem = [x for x in interpolated_df.columns if x not in moving_cols]
    interpolated_df = interpolated_df[moving_cols + rem]
    return interpolated_df

def obtain_interpolated_df():
    """Obtain the interpolated DataFrame with the period column attached."""
    df = get_stacked_person_csvs()
    df = df[['person', 'Session', 'chest_hrv']]
    df = completely_interpolate(df)

    df = attach_period_column(df)
    df = df.dropna()
    return df

def attach_hrv_to_metrics(df, int_df_manual):
    """Attach the mean chest_hrv of each person / session / period to the metrics DataFrame."""
    person = df['person'].iloc[0]
    session = df['session'].iloc[0]
    period = df['period'].iloc[0]

    info = int_df_manual[(int_df_manual['person'] == person) & (int_df_manual['Session'] == session) & (int_df_manual['period'] == period)]['chest_hrv']

    df['mean_chest_hrv'] = info.mean()

    return df

def obtain_metrics_df():
    """Obtain the metrics DataFrame with the mean chest_hrv for each period attached."""
    metrics_pre = stack_people_metrics()
    interpolated_df = obtain_interpolated_df()


    replace_val = metrics_pre['nback_mean_response_time'].mean()

    metrics_pre['nback_mean_response_time'] = metrics_pre['nback_mean_response_time'].fillna(replace_val)

    metrics = metrics_pre.groupby(['person', 'session', 'period'], group_keys=False)[metrics_pre.columns].apply(attach_hrv_to_metrics, interpolated_df)

    # for each row, make a new column of the mean of crt_accuracy and nback_accuracy
    metrics['total_accuracy'] = metrics.apply(lambda x: (x['crt_accuracy'] + x['nback_accuracy']) / 2, axis=1)
    metrics['total_response_time'] = metrics.apply(lambda x: (x['crt_mean_response_time'] + x['nback_mean_response_time']) / 2, axis=1)

    baseline_hrv = interpolated_df[interpolated_df['period'] == 'baseline'].groupby(['person', 'Session'])['chest_hrv'].mean().reset_index()
    baseline_hrv = baseline_hrv.rename(columns={'chest_hrv': 'baseline_hrv', 'Session': 'session'})

    # Merge baseline HRV with metrics
    combined_data = pd.merge(metrics, baseline_hrv, on=['person', 'session'])

    combined_data['hrv_diff'] = combined_data['mean_chest_hrv'] - combined_data['baseline_hrv']

    
    cols_to_move = ['person', 'session', 'period', 'hrv_diff']
    remaining_columns = [col for col in combined_data.columns if col not in cols_to_move]

    combined_data = combined_data[cols_to_move + remaining_columns]

    return combined_data


# ====================
# eeg.ipynb functions
# ====================


def generate_experimental_csvs():
    """Only used for eeg.ipynb, depreciated."""
    #NOTE This compiles person.csv into experimental_person.csv. Add queries to drop certain rows HERE
    os.chdir(REPO_DIRECTORY)
    for person in range(1, 13):
        csvs = []
        for session in ['low', 'medium', 'high']:
            person, session = remap_person_and_session(person, session)
            # Load the data
            path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_data.csv")
            data = pd.read_csv(path)
            data['DateTime'] = pd.to_datetime(data['DateTime'])  # Convert DateTime column if needed
            data.set_index('DateTime', inplace=True)
            data['Faulty'] = 0

            # Load the metadata
            path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_metadata", session, "exp_markers.csv")
            metadata = pd.read_csv(path)
            # Find the start and end of the baseline
            metadata['utcTime'] = metadata['utcTime'].apply(timestamp_to_UTC)

            # Convert the DateTime column to a datetime object
            metadata['utcTime'] = pd.to_datetime(metadata['utcTime'])



            start_baseline = metadata.loc[metadata['eventMarker'] == 'start_baseline', 'utcTime'].values[0]
            end_baseline = metadata.loc[metadata['eventMarker'] == 'end_baseline', 'utcTime'].values[0]


            # Already figured out minimum fatigue_time is 144 seconds, so bump up start fatigue time by like 20 seconds maybe

            # # Find the start and end of the fatigue
            start_fatigue = metadata.loc[metadata['eventMarker'] == 'start_fatigue', 'utcTime'].values[0]

            # COULD BE BENEFICIAL
            start_fatigue += np.timedelta64(20, 's')

            end_fatigue = metadata.loc[metadata['eventMarker'] == 'end_fatigue', 'utcTime'].values[0]



            baseline_data = data[(data.index >= start_baseline) & (data.index <= end_baseline)].copy()

            fatigue_data = data[(data.index >= start_fatigue) & (data.index <= end_fatigue)].copy()


            # # Add the binary encoder
            baseline_data['experimental'] = 0
            fatigue_data['experimental'] = 1

            # ======= Below is the old code that drops columns if the mode is more than 90% of the data =======

            # # Now data is separated into baseline and experimental condition, if the mode of each is more
            # # than 80% of the data, drop the column, as the readings are just so off
            # drop_based_on_mode = True
            # if drop_based_on_mode:
            #     # find AF7 alpha mode
            #     mode = baseline_data['alpha_AF7'].mode()[0]
            #     mode_count = baseline_data['alpha_AF7'].value_counts()[mode]
            #     # If the mode is the more than 90% of the data, drop the column
            #     if mode_count > 0.9 * baseline_data.shape[0]:
            #         print(f'dropping alpha_AF7 with mode {mode} and count {mode_count}')
            #         baseline_data['Faulty'] = 1
            #         fatigue_data['Faulty'] = 1
            #     # display(fatigue_data)
            #     mode = fatigue_data['alpha_AF7'].mode()[0]
            #     mode_count = fatigue_data['alpha_AF7'].value_counts()[mode]
            #     # If the mode is the more than 90% of the data, drop the column
            #     if mode_count > 0.9 * fatigue_data.shape[0]:
            #         print(f'dropping alpha_AF7 with mode {mode} and count {mode_count}')
            #         fatigue_data['Faulty'] = 1
            #         baseline_data['Faulty'] = 1
            # Past stuff sets faulty to 1 if the mode is more than 90% of the data,
            # Either baseline or fatigue data is faulty, so drop it

            # =====================================================================================================


            combined_data = pd.concat([baseline_data, fatigue_data])

            csvs.append(combined_data)

            
        person_experimental_csv = pd.concat(csvs, axis=0)

        path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f"{person}_experimental_data.csv")
        person_experimental_csv.to_csv(path, index=True, encoding='utf-8-sig')

def get_peaks_series(df, prominence):
    """Identifies peaks in an eeg DataFrame and returns the series of the peaks."""
    peak_indices, _ = find_peaks(df, prominence=prominence)

    peak_timestamps = df.index[peak_indices]

    series = df[peak_timestamps]

    return series

def get_data_from_csvs():
    # Full Working Data

    # Compile a csv stacking each person's experimental data
    csvs = []
    for person in range(1, 13):
        person, session = remap_person_and_session(person, 9)


        path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f'{person}_data.csv')


        data = pd.read_csv(path, parse_dates=['DateTime'], index_col='DateTime')
        # add person column
        data['person'] = int(person)
        csvs.append(data)
    full_experimental_data = pd.concat(csvs, axis=0)

    # Change session from object type to string type
    full_experimental_data['Session'] = full_experimental_data['Session'].astype(str)


    # Reorder just for ease of viewing
    columns_to_move = ['person', 'Session']  # specify in the order you want them moved
    new_order = columns_to_move + [col for col in full_experimental_data.columns if col not in columns_to_move]
    full_experimental_data = full_experimental_data[new_order]

    return full_experimental_data

def get_experimental_data():
    # Full Working Data

    # Compile a csv stacking each person's experimental data
    csvs = []
    for person in range(1, 13):
        person, session = remap_person_and_session(person, 9)


        path = os.path.join(Config.FATIGUESET_AGG_DIRECTORY, f'{person}_experimental_data.csv')


        data = pd.read_csv(path, parse_dates=['DateTime'], index_col='DateTime')
        # add person column
        data['person'] = int(person)
        csvs.append(data)
    full_experimental_data = pd.concat(csvs, axis=0)

    # Change session from object type to string type
    full_experimental_data['Session'] = full_experimental_data['Session'].astype(str)


    # Reorder just for ease of viewing
    columns_to_move = ['person', 'Session']  # specify in the order you want them moved
    new_order = columns_to_move + [col for col in full_experimental_data.columns if col not in columns_to_move]
    full_experimental_data = full_experimental_data[new_order]

    return full_experimental_data
