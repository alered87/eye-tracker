"""
    Configuration file to set cleaning parameters


"""


"""
        CLEANING PARAMETER

"""

# # Horizontal boundaries
# X_MIN = 0
# X_MAX = 1024
#
# # Vertical boundaries
# Y_MIN = 0
# Y_MAX = 768
#
# # Fixations Threshold
# F_MIN = 100.0
# F_MAX = 350.0
#
# TRIGGER_DURATION = 100  # In milliseconds
# TAIL_SIZE = 0  # In number of fixations


"""
        COLUMN NAMES

"""

# session = 'RECORDING_SESSION_LABEL'
# duration = 'CURRENT_FIX_DURATION'
# norm_duration = 'NORM_FIX_DURATION'
# area = 'CURRENT_FIX_INTEREST_AREAS'
# x = 'CURRENT_FIX_X'
# y = 'CURRENT_FIX_Y'
# p = 'CURRENT_FIX_PUPIL'
# norm_p = 'NORM_FIX_PUPIL'
# start = 'CURRENT_FIX_START'
# image = 'image'
# exp = 'experiment'
# trial = 'TRIAL_INDEX'
# stroop_type = 'stroop_type'


AREAS_DICT = {
    '[ ]': 0,
    '[ 1]': 1,
    '[ 2]': 2,
    '[ 3]': 3,
    '[ 4]': 4,
    '[ 5]': 5,
    '[ 6]': 6,
    '[ 7]': 7,
    '[ 8]': 8,
    '[ 9]': 9,
    '[ 10]': 10,
    '[ 11]': 11,
    '[ 12]': 12,
    '[ 13]': 13,
    '[ 14]': 14,
    '[ 15]': 15,
    '[ 16]': 16,
    '[ 1, 999]': 99
}

IMAGES_DICT = {
    'NamingWITHinterference.jpg': 'NWI',
    'NamingWITHOUTinterference.jpg': 'NWoI',
    'ReadingWITHinterference.jpg': 'RWI',
    'ReadingWITHOUTinterference.jpg': 'RWoI'
}

TYPE_DICT = {
    'NamingWITHinterference.jpg': 'Naming',
    'NamingWITHOUTinterference.jpg': 'Naming',
    'ReadingWITHinterference.jpg': 'Reading',
    'ReadingWITHOUTinterference.jpg': 'Reading'
}

LABEL_MAP = {
    'NamingWITHinterference.jpg': 0,
    'NamingWITHOUTinterference.jpg': 1,
    'ReadingWITHinterference.jpg': 2,
    'ReadingWITHOUTinterference.jpg': 3
}
