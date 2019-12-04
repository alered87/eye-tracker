import numpy as np
import pandas as pd

from config import *


def clean_fixations(raw_df, x_min=0, x_max=1024, x_name='x',
                    y_min=0, y_max=768, y_name='y',
                    duration_name='duration', fix_min=100.0, fix_max=400.0):
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
    :return:
    """
    # Clean x position
    raw_df = raw_df[(raw_df[x_name] > x_min) & (raw_df[x_name] < x_max)]

    # Clean y position
    raw_df = raw_df[(raw_df[y_name] > y_min) & (raw_df[y_name] < y_max)]

    # Clean Fixations
    raw_df = raw_df[(raw_df[duration_name] > fix_min) & (raw_df[duration_name] < fix_max)]

    return raw_df


def extract_fixations_stats(df, compute_stats=True, compute_areas_stats=True,
                            session_name='session', image='image', area='area', int_area='int_area',
                            start='start', trigger_duration=100.0, tail=5,
                            pupil_size_name='pupil', norm_pupil_size_name='norm_pupil_size',
                            duration='duration', norm_duration='norm_duration'):
    """
    Clean fixations data-frame and compute statistics image-wise [optional] and/or area-wise [optional]

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
        'norm_fix_max',  # Normalized max pupil size
        'norm_fix_mean',  # Normalized average pupil size
        'norm_fix_std'  # Normalized std of pupil size
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
        cdf[norm_duration] = cdf[duration] - cdf[duration].mean()

        for img in cdf[image].unique():

            # Split by image
            idf = cdf[cdf[image] == img]

            # # # Clean head
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
                    idf[norm_pupil_size_name].std()  # Normalized std of pupil size
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
        return_list.append(pd.DataFrame(data=agg_data, columns=new_columns))

    if compute_areas_stats:
        return_list.append(np.nan_to_num(areas_agg_data))

    return return_list
