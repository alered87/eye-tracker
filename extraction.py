import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

from config import *


def compute_outlier_threshold(pandas_series):
    return (pandas_series.quantile(0.75) - pandas_series.quantile(0.25)) * 1.5 + pandas_series.quantile(0.75)


def clean_fixations(raw_df, x_min=0, x_max=1024, x_name='x',
                    y_min=0, y_max=768, y_name='y',
                    duration_name='duration', fix_min=100.0, fix_max=400.0, fix_cleaning='minmax'):
    """
        Clean noisy fixations data depending on x, y boundaries and fixations min-max duration

    :param raw_df:
    :param x_min:
    :param x_max:
    :param x_name:
    :param y_min:
    :param y_max:
    :param y_name:
    :param duration_name:
    :param fix_min:
    :param fix_max:
    :param fix_cleaning:

    :return:
    """
    # Clean x position
    raw_df = raw_df[(raw_df[x_name] > x_min) & (raw_df[x_name] < x_max)]

    # Clean y position
    raw_df = raw_df[(raw_df[y_name] > y_min) & (raw_df[y_name] < y_max)]

    # Clean Fixations
    if fix_cleaning == 'minmax':
        raw_df = raw_df[(raw_df[duration_name] > fix_min) & (raw_df[duration_name] < fix_max)]

    elif fix_cleaning == 'z_score':
        raw_df = raw_df[np.abs(zscore(raw_df[duration_name])) < 3]  # Cleaning by removing outlier

    elif fix_cleaning == 'outliers_iter':
        last_thr = raw_df[duration_name].max()
        c_fix_thr = compute_outlier_threshold(raw_df[duration_name])
        while (last_thr - c_fix_thr) > 0.0:
            raw_df = raw_df[raw_df[duration_name] < c_fix_thr]
            print(last_thr, c_fix_thr)
            last_thr = c_fix_thr
            c_fix_thr = compute_outlier_threshold(raw_df[duration_name])

    return raw_df


def extract_fixations_stats(df, compute_stats=True, compute_areas_stats=True, clean_areas=True,
                            session_name='session', image='image', area='area', int_area='int_area',
                            start='start', trigger_duration=100.0, tail=5, minimum_fixation_length=100.0,
                            pupil_size_name='pupil', norm_pupil_size_name='norm_pupil_size',
                            norm0_pupil_size_name='norm0_pupil_size', x='CURRENT_FIX_X', y='CURRENT_FIX_Y',
                            duration='duration', norm_duration='norm_duration'):
    """
    Clean fixations data-frame and compute statistics image-wise [optional] and/or area-wise [optional]

    :param clean_areas:
    :param y:
    :param x:
    :param minimum_fixation_length:
    :param norm0_pupil_size_name:
    :param df:
    :param compute_stats:
    :param compute_areas_stats:
    :param session_name:
    :param image:
    :param area:
    :param int_area:
    :param start:
    :param trigger_duration:
    :param tail:
    :param pupil_size_name:
    :param norm_pupil_size_name:
    :param duration:
    :param norm_duration:
    :return:
    """
    new_columns = [
        session_name,  # Session
        image,  # Image name (coded)
        # Fixations
        'n_fix',  # Fixations number
        'fix_max',  # Maximum fixations duration
        'fix_mean',  # Average fixations duration
        'fix_std',  # Std fixations duration
        # Normalized Fixations
        'norm_fix_max',  # Normalized maximum fixations duration
        'norm_fix_mean',  # Normalized average fixations duration
        'norm_fix_std',  # Normalized std fixations duration
        # Pupil size
        'pupil_max',  # Maximum pupil size
        'pupil_mean',  # Average pupil size
        'pupil_std',  # Std of pupil size
        # Normalized Pupil Size
        'norm_pupil_max',  # Normalized max pupil size
        'norm_pupil_mean',  # Normalized average pupil size
        'norm_pupil_std',  # Normalized std of pupil size
        # 1st pupil size Normalized Pupil Size
        'norm0_pupil_max',  # Normalized max pupil size
        'norm0_pupil_mean',  # Normalized average pupil size
        'norm0_pupil_std',  # Normalized std of pupil size
        # Total task time
        'task_time',
        'minimum_length',
        'x_regressions',
        'y_regressions'
    ]
    agg_data = []  # Data of all the statistics on cleaned images
    areas_agg_data = []  # Data of all the statistics on cleaned images, computed by area-of-interest
    clean_df = None  # Final Cleaned Dataframe
    fix_start_dict = {}  # Saving start, end time (can be used to filter raw gaze data)
    skipped = {}  # List of skipped empty images

    for s in df[session_name].unique():

        fix_start_dict[s] = {}

        # Split by Session/subject
        cdf = df[df[session_name] == s]

        # Subject-wise Normalization
        cdf[norm_pupil_size_name] = cdf[pupil_size_name] - cdf[pupil_size_name].mean()
        cdf[norm0_pupil_size_name] = cdf[pupil_size_name] - cdf[pupil_size_name].iloc[0]
        cdf[norm_duration] = cdf[duration] - cdf[duration].mean()

        for img in cdf[image].unique():

            # Split by image
            idf = cdf[cdf[image] == img]

            # # # Clean head
            if trigger_duration > 0:
                idf[int_area] = idf[area].apply(lambda x: AREAS_DICT[x])  # Convert areas names in int
                idf['trigger'] = idf[int_area] == 99  # Trigger area detection

                # Detect first fixation after trigger
                start_ind = idf[[all([a, b]) for a, b in zip(idf['trigger'], idf[duration] > trigger_duration)]].index.min()

                # Skipping too short trial
                if len(idf.loc[start_ind:]) <= 1:

                    fix_start_dict[s][img] = -1
                    print(" >>>>> SKIPPING: {}{}".format(s, img))
                    if s in skipped.keys():
                        skipped[s].append(img)
                    else:
                        skipped[s] = [img]
                    continue

                start_loc = idf.index.get_loc(start_ind) + 1  # Compute initial index

                # Saving start, end time (can be used to filter raw gaze data)
                fix_start_dict[s][img] = (idf[start].iloc[start_loc], idf[start].iloc[-tail])
                idf = idf.iloc[start_loc:]  # Selecting final data

            # Clean too short fixations
            idf = idf[idf[duration] > minimum_fixation_length]

            if clean_areas:
                # Clean Interest areas
                idf = idf[idf[int_area].isin([
                    1, 5, 9, 13, 99,  # 1st Row
                    2, 6, 10, 14,  # 2nd Row
                    3, 7, 11, 15,  # 3nd Row
                    4, 8, 12, 16  # 4th Row
                ])]

            # Append data on global DataFrame
            if len(idf) == 0:
                print(" >>>>> LEN SKIPPING: {} - {}".format(s, img))
                if s in skipped.keys():
                    skipped[s].append(img)
                else:
                    skipped[s] = [img]
                continue

            if clean_df is None:
                clean_df = idf.copy()
            else:
                clean_df = clean_df.append(idf)

            # # # Computing statistics on current (c_) df

            # # Global Fixations
            if compute_stats:
                # Fixations
                new_data = [
                    s,  # Session
                    IMAGES_DICT[img],  # Image name (coded)
                    # Fixations
                    len(idf),  # Fixations number
                    idf[duration].max(),  # Maximum fixations duration
                    idf[duration].mean(),  # Average fixations duration
                    idf[duration].std(),  # Std fixations duration
                    # Normalized Fixations
                    idf[norm_duration].max(),  # Normalized maximum fixations duration
                    idf[norm_duration].mean(),  # Normalized average fixations duration
                    idf[norm_duration].std(),  # Normalized std fixations duration
                    # Pupil size
                    idf[pupil_size_name].max(),  # Maximum pupil size
                    idf[pupil_size_name].mean(),  # Average pupil size
                    idf[pupil_size_name].std(),  # Std of pupil size
                    # Normalized Pupil Size
                    idf[norm_pupil_size_name].max(),  # Normalized max pupil size
                    idf[norm_pupil_size_name].mean(),  # Normalized average pupil size
                    idf[norm_pupil_size_name].std(),  # Normalized std of pupil size
                    # 1st pupil size Normalized Pupil Size
                    idf[norm0_pupil_size_name].max(),  # Normalized max pupil size
                    idf[norm0_pupil_size_name].mean(),  # Normalized average pupil size
                    idf[norm0_pupil_size_name].std(),  # Normalized std of pupil size

                    # Total task
                    idf.iloc[-1][duration] + idf.iloc[-1][start] - idf.iloc[0][start],

                    minimum_fixation_length,

                    np.sum([-300 < (idf[x].iloc[i] - idf[x].iloc[i - 1]) < 0 for i in range(1, len(idf))]),
                    np.sum([(idf[y].iloc[i] - idf[y].iloc[i - 1]) < 0 for i in range(1, len(idf))])

                ]

                agg_data.append(new_data)

            # # Fixations by area-of-interest
            if compute_areas_stats:
                # Remove area 99
                idf[int_area] = idf[int_area].apply(lambda x: 1 if x == 99 else x)

                c_data = []
                # Stats on areas
                for i in range(1, 17):
                    area_mean = idf[idf[int_area] == i][duration].mean()
                    area_count = idf[idf[int_area] == i][duration].count()
                    area_max = idf[idf[int_area] == i][duration].max()
                    area_min = idf[idf[int_area] == i][duration].min()
                    c_data += [area_mean, area_count, area_min, area_max]

                c_data += [LABEL_MAP[img]]
                areas_agg_data.append(c_data)

    return_list = [clean_df, fix_start_dict, skipped]

    if compute_stats:
        clean_stats = pd.DataFrame(data=agg_data, columns=new_columns)
        clean_stats[image].hist()
        plt.show()
        return_list.append(clean_stats)

    if compute_areas_stats:
        return_list.append(np.nan_to_num(areas_agg_data))

    return return_list


def saccade_distance(row, x_e, x_s, y_e, y_s):
    return np.sqrt((row[x_e]-row[x_s])**2 + (row[y_e]-row[y_s])**2)


def saccade_slope(row, x_e, x_s, y_e, y_s):
    return (row[y_e]-row[y_s])/(row[x_e]-row[x_s])


def saccade_stats(saccades_df, image='image'):

    new_columns = [
        image,
        # Saccade Direction counting
        'up_freq', 'down_freq', 'left_freq', 'right_freq',
        # Blink counting and stats
        'n_blink', 'min_blink', 'avg_blink', 'max_blink',

    ]
    for c_var in ['duration', 'vel', 'ampl', 'angle', 'distance', 'slope']:
        for c_measure in ['min', 'avg', 'max']:#, 'std']:
            new_columns.append("{0}_{1}".format(c_measure, c_var))

    x_start, y_start = 'CURRENT_SAC_START_X', 'CURRENT_SAC_START_Y'
    x_end, y_end = 'CURRENT_SAC_END_X', 'CURRENT_SAC_END_Y'
    angle = 'CURRENT_SAC_ANGLE'
    ampl = 'CURRENT_SAC_AMPLITUDE'
    vel = 'CURRENT_SAC_AVG_VELOCITY'
    duration = 'CURRENT_SAC_DURATION'
    blink_duration = 'CURRENT_SAC_BLINK_DURATION'
    area = 'CURRENT_SAC_END_INTEREST_AREA_ID'
    direction = 'CURRENT_SAC_DIRECTION'
    distance = 'distance'
    slope = 'slope'

    float_vars = [x_start, y_start, x_end, y_end, angle, ampl, vel, duration]
    int_vars = [area]
    clean_vars = [direction]

    gdf = saccades_df.copy()
    gdf[blink_duration] = gdf[blink_duration].apply(lambda x: '0' if x == '.' else x).astype(int)

    # Cleaning empty values
    for c_var in clean_vars + float_vars + int_vars:
        gdf = gdf[gdf[c_var] != '.']

    for c_var in float_vars:
        gdf[c_var] = gdf[c_var].astype(float)

    for c_var in int_vars:
        gdf[c_var] = gdf[c_var].astype(int)

    gdf[distance] = gdf.apply(saccade_distance, args=(x_end, x_start, y_end, y_start), axis=1)
    gdf[slope] = gdf.apply(saccade_slope, args=(x_end, x_start, y_end, y_start), axis=1)

    all_data = []
    for img in gdf[image].unique():

        new_data = [IMAGES_DICT[img]]
        # Selecting current image
        df = gdf[gdf[image] == img]

        # Saccades Directions counting
        direction_count = df.groupby([direction])['experiment'].count()
        for c_direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if c_direction in direction_count:
                new_data.append(direction_count[c_direction]/direction_count.sum())
            else:
                new_data.append(0)

        # Blink counting and duration
        blink_df = df[df[blink_duration] != 0]
        new_data.append(len(blink_df))

        new_row = [0, 0, 0]
        if len(blink_df) > 0:
            new_row = [
                blink_df[blink_duration].min(),
                blink_df[blink_duration].mean(),
                blink_df[blink_duration].max()
            ]

        new_data += new_row

        for stat_var in [duration, vel, ampl, angle, distance, slope]:
            new_data += [df[stat_var].min(), df[stat_var].mean(), df[stat_var].max()]#, df[stat_var].std()]

        all_data.append(new_data)

    return pd.DataFrame(data=all_data, columns=new_columns)


