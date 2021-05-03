"""
Author: Hannes Kr. Hannesson

Purpose of this script is to compare manual scoring of spindles to automatically found ones from YASA
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd
import yasa
import h5py
import yasa
from pprint import pprint
from dateutil.rrule import rrule, SECONDLY
import bisect
from datetime import timedelta
import seaborn as sns
from loguru import logger

import argparse

from models import Signal


parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="count", default=0)
parser.add_argument("--multi_channel", action="store_true", help="Use all EEG channels for spindle detection")
parser.add_argument("--plot_matrix", action="store_true", help="Plot a heatmap of real vs predicted pairs")
parser.add_argument("--plot_pairs", action="store_true", help="Plot a comparison between yasa and expert")
parser.add_argument("--ovt", default=0.2, type=float, help="Overlap threshold, between 0.0 and 1.0 (both inclusive)")
parser.add_argument("--n2_only", action="store_true", help="Validation will only involve looking at epochs which have been scored as N2 (default is N1, N2, and N3)")
parser.add_argument("--all_but_n2", action="store_true", help="Validation will only involve looking at epochs which have not been scored as N2 (default is N1, N2, and N3)")
parser.add_argument("--all_stages", action="store_true", help="Validation will involve looking at epochs regardless of sleep stage (default is N1, N2, and N3)")
parser.add_argument("--channel", dest="chosen_channel", choices = ["C", "F"], help="Determines the channel that YASA will analyze (if --multi_channel mode is not activated), also determines what expert scored spindles we will compare to (regardless of whether --multi_channel is on)")
parser.add_argument("--remove_outliers", action="store_true", help="Use an isolation forest algorithm to reject outlying spindles")
parser.add_argument("--multi_only", action="store_true", help="Determines whether a spindle needs to be detected on 2 or more channels to be considered")

args = parser.parse_args()

MULTI_CHANNEL = args.multi_channel
PLOT_MATRIX = args.plot_matrix
PLOT_PAIRS = args.plot_pairs
OVT = args.ovt
N2_ONLY = args.n2_only
CHOSEN_CHANNEL = args.chosen_channel
REMOVE_OUTLIERS = args.remove_outliers
MULTI_ONLY = args.multi_only

stage_flag_count = 0

if args.n2_only:
    stage_flag_count += 1 
if args.all_but_n2:
    stage_flag_count += 1
if args.all_stages:
    stage_flag_count += 1

if stage_flag_count > 1:
    print("Illegal combinaion of arguments, only one sleep stage modifier can be chosen at a time")
    exit(1)

if not CHOSEN_CHANNEL:
    print("You need to supply a channel, --help for help")
    exit(1)

from utils import (
    read_scored_desat_indices,
    read_scored_desat_indices_as_bool_vector,
    get_desat_aux_signal_from_scores,
    get_spindles_aux_signal_from_eeg_signal,
    detect_spindles,
    get_hypno_list_from_scores,
    plot_spindles,
    plot_o2,
    plot_specta,
    plot_welch,
    plot_stages,
    range_overlap,
    range_union,
    compute_max_pairs,
    convert_str_to_datetime,
    range_overlap_coefficient,
    apply_noxturnal_filtering,
    apply_powerline_filtering_only,
    combine_spindles_across_channels,
    combine_spindles_across_channels_yasa_style,
    ScoringKeyException,
    get_range_between_first_epoch_and_last_epoch_onsets_over_scores,
    get_range_between_first_epoch_and_last_epoch_onsets_over_signal,
    get_range_between_first_epoch_and_last_epoch_onsets,
    eeg_sampling_frequencies,
    resamplify_dataframe,
    get_duration_of_start_stop_range
)

logger.add(f"spindle_verification.log", format = "{time} {level} | {file}:{line}({function}): <level>{message}</level>", level="WARNING", backtrace=True, rotation="10 GB")

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

def auto_get_ranges_from_csv(csv_path, manual_scores):
    #first_epoch_start, _ = get_range_between_first_epoch_and_last_epoch_onsets(manual_scores)
    first_epoch_start = 0
    df = pd.read_csv(csv_path)
    ranges = []
    already_seen = set()
    for _, row in df.iterrows():
        epoch_beginning = row["Epoch start (seconds)"]
        if epoch_beginning not in already_seen:
            ranges.append((epoch_beginning, epoch_beginning+30))
            already_seen.add(epoch_beginning)

    # make another pass, subtract the part before the first scored epoch
    #adjusted_ranges = []
    #for start, end in ranges:
    #    adjusted_ranges.append((start-first_epoch_start, end-first_epoch_start))
    #return adjusted_ranges
    return ranges

def print_spindles_dataframe_with_adjusted_onset(csv_path, manual_scores):
    first_epoch_start, _ = get_range_between_first_epoch_and_last_epoch_onsets(manual_scores)
    df = pd.read_csv(csv_path)
    df["Onset seconds"] += first_epoch_start
    print("Printing dataframe of manually scored spindles, with adjusted onsets based on first scored epoch")
    print(df)

    
def read_json(path):
    scores = None
    with open(path) as f:
        scores = json.load(f)
    return scores



info = {
    #"VSN_14_080_001_epoch_series": {
    #    "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_001.hdf5",
    #    "scoring_file": "/home/hannes/vsn-14-080-derived/spindles_larger_segments/spindles_001new.csv",
    #    "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/spindles_larger_segments/spindles_001new.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json")),
    #    "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json")
    #},
    #"VSN_14_080_007_epoch_series": {
    #    "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_007.hdf5",
    #    "scoring_file": "/home/hannes/vsn-14-080-derived/spindles_larger_segments/spindles_007new.csv",
    #    "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/spindles_larger_segments/spindles_007new.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json")),
    #    "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json")
    #},
    # "VSN_14_080_001_individual_epochs": {
    #     "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_001.hdf5",
    #     "scoring_file": "/home/hannes/vsn-14-080-derived/spindles_single_epochs/spindles_001.csv",
    #     "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/spindles_single_epochs/spindles_001.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json")),
    #     "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json")
    # },
    # "VSN_14_080_003_individual_epochs": {
    #     "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_003.hdf5",
    #     "scoring_file": "/home/hannes/vsn-14-080-derived/spindles_single_epochs/spindles_003.csv",
    #     "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/spindles_single_epochs/spindles_003.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_003.scoring.json")),
    #     "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_003.scoring.json")
    # },
    # "VSN_14_080_007_individual_epochs": {
    #     "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_007.hdf5",
    #     "scoring_file": "/home/hannes/vsn-14-080-derived/spindles_single_epochs/spindles_007.csv",
    #     "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/spindles_single_epochs/spindles_007.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json")),
    #     "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json")
    # }
    "VSN_14_080_001_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_001.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_001_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_001_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_004_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_004.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_004_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_004_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_004.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_004.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_006_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_006.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_006_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_006_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_006.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_006.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_007_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_007.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_007_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_007_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_008_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_008.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_008_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_008_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_008.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_008.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_009_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_009.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_009_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_009_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_009.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_009.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_010_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_010.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_010_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_010_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_010.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_010.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_011_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_011.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_011_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_011_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_011.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_011.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_012_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_012.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_012_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_012_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_012.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_012.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_013_C_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_013.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/C_only/Spindle report_013_C.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/C_only/Spindle report_013_C.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_013.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_013.scoring.json"),
         "channel": "C"
    },
    "VSN_14_080_001_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_001.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_001_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_001_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_001.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_004_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_004.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_004_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_004_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_004.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_004.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_006_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_006.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_006_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_006_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_006.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_006.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_007_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_007.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_007_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_007_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_007.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_008_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_008.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_008_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_008_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_008.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_008.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_009_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_009.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_009_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_009_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_009.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_009.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_010_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_010.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_010_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_010_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_010.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_010.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_011_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_011.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_011_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_011_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_011.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_011.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_012_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_012.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_012_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_012_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_012.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_012.scoring.json"),
         "channel": "F"
    },
    "VSN_14_080_013_F_only" : {
        "data_file": "/datasets/VSN-14-080/HDF5/VSN_14_080_013.hdf5",
         "scoring_file": "/home/hannes/vsn-14-080-derived/F_only/Spindle report_013_F.csv",
         "scoring_seconds_ranges": auto_get_ranges_from_csv("/home/hannes/vsn-14-080-derived/F_only/Spindle report_013_F.csv", read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_013.scoring.json")),
         "manual_scoring_file": read_json("/datasets/VSN-14-080/HDF5/VSN_14_080_013.scoring.json"),
         "channel": "F"
    },
}

def cut_into_segments(arr, scoring_seconds_ranges, sf):
    """
    Args:
        arr (2d numpy arr): [description]
        scoring_seconds_ranges ([type]): [description]
        sf (int): sampling frequency
    """
    segments = []
    for start, end in scoring_seconds_ranges:
        start_sample = int(sf * start)
        end_sample = int(sf * end)

        segment = arr[:, start_sample:end_sample]
        yield segment 
        # as yasa returns a 2d array when given multiple EEG channels, we must support that here
        #segments.append(arr[:, start_sample:end_sample])

    #segments

def count_intersections(lis1, lis2):
    # finds the argmax of the lists, where the function is the length, i.e. finds the shortest list
    shorter_list = min([lis1, lis2], key=lambda x: len(x))
    intersections_count = 0
    for i in range(shorter_list):
        if range_overlap(lis1[i], lis2[i]) > 0:
            intersections_count += 1

def round_timestamp_down_to_nearest_epoch(ts):
    """Given a timestamp, rounds it down to the nearest timestamp that ends in hh:mm:00 or hh:mm:30 (00 seconds or 30 seconds)

    Functionality is based on stackoverflow answers (however edited to fit my use case): https://stackoverflow.com/a/32723408, https://stackoverflow.com/a/41595868

    Args:
        ts (datetime): Timestamp you wish to round down

    Returns:
        datetime: downrounded timestamp
    """
    # Round down to nearest hour
    rounded_down_hour_ts = ts.replace(microsecond=0, second=0, minute=0)
    # 120 epochs, as there are 60 mins in 1 hour, and 2 epochs in every minute, we get 60*2 = 120, one extra to make it exactly one hour
    rounds = list(rrule(SECONDLY, interval=30, count=121, dtstart=rounded_down_hour_ts))
    closest_30secs_ts = rounds[bisect.bisect(rounds, ts)]
    return closest_30secs_ts


def get_epochs(starting_ts, epoch_count, convert_to_seconds=False):
    rounded_ts = round_timestamp_down_to_nearest_epoch(starting_ts)
    epochs = list(rrule(SECONDLY, interval=30, count=epoch_count, dtstart=rounded_ts))

    epochs_ranges = []

    for epoch in epochs:
        epochs_ranges.append((epoch, epoch + timedelta(seconds=30)))

    if convert_to_seconds:
        convert_to_second_func = lambda x: (x - starting_ts).total_seconds()
        starts, ends = zip(*epochs_ranges)
        epochs_ranges = zip(map(convert_to_second_func, starts), map(convert_to_second_func, ends))
    return epochs_ranges

def evaluate_pairs(pairs):
    range_overlap_coefficient_list = []
    for score, range1, range2 in pairs:
        range_overlap_coefficient_value = range_overlap_coefficient(range1, range2)
        range_overlap_coefficient_list.append(range_overlap_coefficient_value)

    try:
        avg_range_overlap_coefficient = sum(range_overlap_coefficient_list) / len(range_overlap_coefficient_list)
    except ZeroDivisionError:
        logger.warning("Zero divion error when computing overlap coefficient, returning 0")
        avg_range_overlap_coefficient = 0
    logger.info(f"Average overlap coefficient: {avg_range_overlap_coefficient:.2f}")

def get_range_between_first_epoch_and_last_epoch_onsets_over_spindle_scoring(manual_scores, spindle_scores):
    start, end = get_range_between_first_epoch_and_last_epoch_onsets(manual_scores)
    #print(spindle_scores)
    cols_to_keep = ["Onset seconds", "Duration"]
    #spindle_scores_copy = spindle_scores.drop(columns=["Epoch start (seconds)", "Start Time", "Recording start"])
    spindle_scores_copy = spindle_scores.drop(spindle_scores.columns.difference(cols_to_keep), axis=1)
    spindle_scores_copy["Onset seconds"] = spindle_scores_copy["Onset seconds"].apply(lambda x: x - start)
    #print(spindle_scores_copy)
    return spindle_scores_copy
    

def verify():
    all_recalls = {}
    all_precisions = {}
    all_f1s = {}
    all_TPs = {}
    all_FPs = {}
    all_FNs = {}
    for panel in info:
        #DEBUG, REMOVE WHEN DONE
        #if "012" not in panel:
        #    continue
        data_path = info[panel]["data_file"]
        channel = info[panel]["channel"]
        if CHOSEN_CHANNEL != channel:
            continue
        data_file = h5py.File(data_path)
        spindle_scoring_path = info[panel]["scoring_file"]
        logger.debug("data file: %s" % data_path)
        logger.debug("spindle scoring file: %s" % spindle_scoring_path)
        logger.info("Procesing panel: %s" % panel)
        spindle_scores = pd.read_csv(spindle_scoring_path)

        panel_name = info[panel]["data_file"].split("/")[-1].split(".")[0]
        sampling_frequency = eeg_sampling_frequencies[panel_name]
        spindle_scores = resamplify_dataframe(spindle_scores, sampling_frequency, 200)





        # the manual scores from the json file, contains various events, not to mention the sleep stages
        # manual_scores is not readjusted for the window
        manual_scores = info[panel]["manual_scoring_file"]

        period_analysis_between_first_and_last_epoch_start, period_analysis_between_first_and_last_epoch_stop = get_range_between_first_epoch_and_last_epoch_onsets(manual_scores)

        #logger.debug(f"length of period analysis: {str(timedelta(seconds=(period_analysis_between_first_and_last_epoch_stop-period_analysis_between_first_and_last_epoch_start)))}")
        #logger.debug(f"length of period analysis in seconds: {period_analysis_between_first_and_last_epoch_stop-period_analysis_between_first_and_last_epoch_start}")
        max_onset_event = spindle_scores["Onset seconds"].max()

        # cull out any expert spindles that occur outside the window that we analyze
        spindle_scores = spindle_scores[(spindle_scores["Onset seconds"] >= period_analysis_between_first_and_last_epoch_start) & ((spindle_scores["Onset seconds"] + spindle_scores["Duration"]) < (period_analysis_between_first_and_last_epoch_stop + 30))]


        

        spindle_scores = get_range_between_first_epoch_and_last_epoch_onsets_over_spindle_scoring(manual_scores, spindle_scores)
        #print_spindles_dataframe_with_adjusted_onset(info[panel]["scoring_file"], info[panel]["manual_scoring_file"])

        


        scoring_seconds_ranges = info[panel]["scoring_seconds_ranges"]
        # need to offset the scoring second ranges, so it matches the 'cut' signal (i.e. after cutting the signal to be between first and last epoch of the period analysis)
        scoring_seconds_ranges = [(start, stop) for start, stop in scoring_seconds_ranges if start >= period_analysis_between_first_and_last_epoch_start and stop < period_analysis_between_first_and_last_epoch_stop]
        first_scoring_range_start = scoring_seconds_ranges[0][0]
        scoring_seconds_ranges = [(start-first_scoring_range_start, stop-first_scoring_range_start) for start, stop in scoring_seconds_ranges]

        # rounding, to avoid float issues
        #scoring_seconds_ranges = [(round(start), round(stop)) for start, stop in scoring_seconds_ranges]
        epoch_scores = []

        eeg_c3_m2 = Signal(data_file["C3-M2"][()], "C3-M2", data_file["C3-M2"].attrs["fs"])
        eeg_f3_m2 = Signal(data_file["F3-M2"][()], "F3-M2", data_file["F3-M2"].attrs["fs"])
        eeg_c4_m1 = Signal(data_file["C4-M1"][()], "C4-M1", data_file["C4-M1"].attrs["fs"])
        eeg_f4_m1 = Signal(data_file["F4-M1"][()], "F4-M1", data_file["F4-M1"].attrs["fs"])
        eeg_o2_m1 = Signal(data_file["O2-M1"][()], "O2-M1", data_file["O2-M1"].attrs["fs"])
        eeg_o1_m2 = Signal(data_file["O1-M2"][()], "O1-M2", data_file["O1-M2"].attrs["fs"])

        # convert to be of uv scale, yasa requires this
        eeg_c3_m2.data *= 1000000
        eeg_f3_m2.data *= 1000000
        eeg_c4_m1.data *= 1000000
        eeg_f4_m1.data *= 1000000
        eeg_o2_m1.data *= 1000000
        eeg_o1_m2.data *= 1000000

        eeg_c3_m2_unfiltered = eeg_c3_m2
        eeg_f3_m2_unfiltered = eeg_f3_m2 
        eeg_c4_m1_unfiltered = eeg_c4_m1
        eeg_f4_m1_unfiltered = eeg_f4_m1
        eeg_o2_m1_unfiltered = eeg_o2_m1
        eeg_o1_m2_unfiltered = eeg_o2_m1

        # No need to filter this, as YASA does a bandpass filter, only need to do powerline filtering
        #eeg_c3_m2 = apply_noxturnal_filtering(eeg_c3_m2)
        #eeg_f3_m2 = apply_noxturnal_filtering(eeg_f3_m2)
        #eeg_c4_m1 = apply_noxturnal_filtering(eeg_c4_m1)
        #eeg_f4_m1 = apply_noxturnal_filtering(eeg_f4_m1)
        #eeg_o2_m1 = apply_noxturnal_filtering(eeg_o2_m1)
        #eeg_o1_m2 = apply_noxturnal_filtering(eeg_o1_m2)

        eeg_c3_m2 = apply_powerline_filtering_only(eeg_c3_m2, verbose=args.verbose)
        eeg_f3_m2 = apply_powerline_filtering_only(eeg_f3_m2, verbose=args.verbose)
        eeg_c4_m1 = apply_powerline_filtering_only(eeg_c4_m1, verbose=args.verbose)
        eeg_f4_m1 = apply_powerline_filtering_only(eeg_f4_m1, verbose=args.verbose)
        eeg_o2_m1 = apply_powerline_filtering_only(eeg_o2_m1, verbose=args.verbose)
        eeg_o1_m2 = apply_powerline_filtering_only(eeg_o1_m2, verbose=args.verbose)

        range_start, range_end = get_range_between_first_epoch_and_last_epoch_onsets(manual_scores)
        #logger.debug(f"range_start: {range_start}")
        #logger.debug(f"range_end: {range_end}")


        #pprint(spindle_scores)

        eeg_c3_m2 = get_range_between_first_epoch_and_last_epoch_onsets_over_signal(eeg_c3_m2, manual_scores)
        eeg_f3_m2 = get_range_between_first_epoch_and_last_epoch_onsets_over_signal(eeg_f3_m2, manual_scores)
        eeg_c4_m1 = get_range_between_first_epoch_and_last_epoch_onsets_over_signal(eeg_c4_m1, manual_scores)
        eeg_f4_m1 = get_range_between_first_epoch_and_last_epoch_onsets_over_signal(eeg_f4_m1, manual_scores)
        eeg_o2_m1 = get_range_between_first_epoch_and_last_epoch_onsets_over_signal(eeg_o2_m1, manual_scores)
        eeg_o1_m2 = get_range_between_first_epoch_and_last_epoch_onsets_over_signal(eeg_o1_m2, manual_scores)

        signal_to_use = eeg_c3_m2
        if CHOSEN_CHANNEL == "F":
            signal_to_use = eeg_f3_m2

        eeg_payload = None
        if not MULTI_CHANNEL:
            # we do this, as cut_into_segments expects a 2d numpy array
            eeg_payload = np.expand_dims((signal_to_use.data), 0)
            channel_names = [signal_to_use.name]
        else:
            eeg_payload = np.vstack((eeg_c3_m2.data, eeg_f3_m2.data, eeg_c4_m1.data, eeg_f4_m1.data, eeg_o2_m1.data, eeg_o1_m2.data))
            channel_names = [eeg_c3_m2.name, eeg_f3_m2.name, eeg_c4_m1.name, eeg_f4_m1.name, eeg_o2_m1.name, eeg_o1_m2.name]



        try:
            hypno_ranges, hypno_list = get_hypno_list_from_scores(manual_scores)
            hypno_range_lookup_by_start_second = {round(start): stage for stage, start, stop in hypno_ranges}
            
        except ScoringKeyException as e:
            logger.warning("%s: Skipping panel" % e)
            continue

        hypno_list_upsampled = yasa.hypno_upsample_to_data(hypno_list, 1/30, signal_to_use.data, signal_to_use.sampling_frequency)

        best_matching_pairs_global = []
        all_ground_truth_spindles = []
        all_yasa_detected_spindles = []
        minimum_distance = 500
        stages_to_include = (1, 2, 3) # YASA default is N1, N2, and N3
        if N2_ONLY:
            stages_to_include = (2,)
        elif args.all_but_n2:
            stages_to_include = (0, 1, 3, 4)
        elif args.all_stages:
            stages_to_include = (0, 1, 2, 3, 4)
                                #W,N1,N2,N3,REM
            

        sp = yasa.spindles_detect(eeg_payload, eeg_f4_m1.sampling_frequency, ch_names=channel_names, multi_only=MULTI_ONLY, remove_outliers=REMOVE_OUTLIERS, min_distance=minimum_distance, verbose=(args.verbose >= 1), hypno=hypno_list_upsampled, include=stages_to_include, duration=(0.5, 2.5))
        yasa_detected_spindles = combine_spindles_across_channels(sp, min_distance=minimum_distance, to_list=True)

        scoring_second_ranges = []
        already_processed_manual_scores = set()
        already_processed_yasa_scores = set()


        #for (start, end), segment in zip(scoring_seconds_ranges, cut_into_segments(eeg_payload, scoring_seconds_ranges, eeg_f3_m2.sampling_frequency)):
        for hypno_stage, start, end in hypno_ranges:
            # we use pandas queries to find samples that belong in the epoch that we are looking at
            #logger.info(f"processing from {start}s to {end}s")
            # If the epoch that we're processing is of a type we dont we dont want to process, then skip ths epoch

            if hypno_stage not in stages_to_include:
            #     logger.debug(f"epoch stage: {hypno_range_lookup_by_start_second[round(start)]}, while only including {', '.join(map(str, stages_to_include))}")
                continue

            # there are 3 cases we want to support:
            # Case 1: the event is wholly contained within @start and @stop, then we're good.
            #        logic: epoch start < event start and event stop < epoch stop
            # Case 2: start point is outside epoch (i.e. before @start), but the other end point is contained within.
            #        logic: event start < epoch start and (epoch start < event stop < epoch stop)
            #            =  event start < epoch start and (epoch start < event stop and event stop < epoch stop)
            # Case 3: end point is outside epoch (i.e. after @stop), but the other end point is contained within.
            #        logic: (epoch start < event start < epoch stop) and epoch stop < event stop
            #             = (epoch start < event start and event start < epoch stop) and epoch stop < event stop
            # This solves the problem where we were discounting spindles that crossed epoch boundaries
            start_manual = "`Onset seconds`"
            stop_manual = "`Onset seconds` + `Duration`"
            case_1_manual = f"@start < {start_manual} and {stop_manual} < @end"
            case_2_manual = f"{start_manual} < @start and (@start < {stop_manual} and {stop_manual} < @end)" 
            case_3_manual = f"(@start < {start_manual} and {start_manual} < @end) and @end < {stop_manual}"
            duration_check_manual = f"`Duration` <= 2.5 and `Duration` >= 0.5"
            #current_epoch_dataframe = spindle_scores.query(f"{case_1_manual} or not {case_1_manual}")
            #current_epoch_dataframe = spindle_scores.query(f"({case_1_manual} or {case_2_manual} or {case_3_manual}) and {duration_check_manual}")
            current_epoch_dataframe = spindle_scores.query(f"{case_1_manual} or {case_2_manual} or {case_3_manual}")

            #current_epoch_dataframe = spindle_scores.query("(@start <= `Onset seconds` and (`Onset seconds` + `Duration`) <= @end) or (`Onset seconds` <= @start) and (`Onset seconds` + `Duration`) <= (@end)")
            # Here we go through the manually scores spindles that are within this epoch (and slightly outside, see cases above), and remove it for consideration for future epochs (as we dont want to double count events)
            ground_truth_spindles = []
            for _, row in current_epoch_dataframe.iterrows():
                if (row["Onset seconds"], row["Onset seconds"] + row["Duration"]) not in already_processed_manual_scores:
                    ground_truth_spindles.append((row["Onset seconds"], row["Onset seconds"] + row["Duration"]))
                    already_processed_manual_scores.add((row["Onset seconds"], row["Onset seconds"] + row["Duration"]))
                    #logger.debug(f"Onset seconds: {row['Onset seconds']}, datetime: {recording_start_period + timedelta(seconds=row['Onset seconds'])}")
                    #logger.debug(f"Onset seconds: {row['Onset seconds']}")


            #yasa_detected_spindles = sp.summary()[["Start", "End"]]
            #yasa_detected_spindles = yasa_detected_spindles.to_records(index=False).tolist()
            #for x, y in yasa_detected_spindles:
            #    if start <= x and y <= end:
            #        print(x, y)
                    
            # here we apply the same cases as outlined above, necessary to hold both manually scored spindles and YASA to the same standard.
            case_1_yasa = lambda start_yasa, stop_yasa: (start < start_yasa) and (stop_yasa < end)
            case_2_yasa = lambda start_yasa, stop_yasa: (start_yasa < start) and (start < stop_yasa and stop_yasa < end)
            case_3_yasa = lambda start_yasa, stop_yasa: (start < start_yasa and start_yasa < end) and end < stop_yasa
            length_check_yasa = lambda start_yasa, stop_yasa: (stop_yasa - start_yasa) <= 2.5 and (stop_yasa - start_yasa) >= 0.5
            current_yasa_detected_spindles = [(yasa_start, yasa_stop) for yasa_start, yasa_stop in yasa_detected_spindles if (case_1_yasa(yasa_start, yasa_stop) or case_2_yasa(yasa_start, yasa_stop) or case_3_yasa(yasa_start, yasa_stop)) and length_check_yasa(yasa_start, yasa_stop)]

            # Only select yasa events that we havent processed before, similar to what we're doing for the manual scores
            current_yasa_detected_spindles = [event for event in current_yasa_detected_spindles if event not in already_processed_yasa_scores]

            #best_matching_pairs, _, _ = compute_max_pairs(ground_truth_spindles, yasa_detected_spindles, overlap_metric)
            #best_matching_pairs_global.extend(best_matching_pairs)
            all_ground_truth_spindles.extend(ground_truth_spindles)
            all_yasa_detected_spindles.extend(current_yasa_detected_spindles)
        
        def plot_pairs():
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex="all", sharey="all")
            seconds_count = np.linspace(start=0, stop=len(eeg_c3_m2)/200, num=len(eeg_c3_m2))
            #datetime_seconds_count = [recording_start_period + timedelta(seconds=s) for s in seconds_count]
            #ax1.plot(seconds_count, eeg_f4_m1.data)
            #ax2.plot(seconds_count, eeg_f4_m1.data)
            ax1.title.set_text("Expert scored spindles")
            ax2.title.set_text("YASA detected spindles")
            ax1.set_ylabel("Voltage (uV)")
            ax2.set_ylabel("Voltage (uV)")
            ax1.set_xlabel("Time (seconds since start)")
            ax2.set_xlabel("Time (seconds since start)")
            #ax1.get_xaxis().set_visible(False)
            #ax2.get_xaxis().set_visible(False)
            filtered_eeg_c3_m2 = apply_noxturnal_filtering(eeg_c3_m2)
            ax1.plot(seconds_count, filtered_eeg_c3_m2.data)
            ax2.plot(seconds_count, filtered_eeg_c3_m2.data)

            for manual_start, manual_stop in all_ground_truth_spindles:
                ax1.axvspan(manual_start, manual_stop, alpha=0.5, color="green")
            for yasa_start, yasa_stop in all_yasa_detected_spindles:
                ax2.axvspan(yasa_start, yasa_stop, alpha=0.5, color="purple")
            epochs_marked = 0
            for (start, end) in scoring_seconds_ranges:
                ax1.axvspan(start, end, alpha=0.3, color="cyan")
                ax2.axvspan(start, end, alpha=0.3, color="cyan")
                epochs_marked += 1
            logger.info(f"Marked {epochs_marked} epochs as blue")
            #plt.suptitle(data_path)
        def plot_unfiltered_vs_filtered_eeg():
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(nrows=6, ncols=2, figsize=(5, 2), sharex="all", sharey="all")
            plot_specta(ax1, eeg_c3_m2)
            plot_specta(ax2, eeg_c3_m2_unfiltered)
            plot_specta(ax3, eeg_c4_m1)
            plot_specta(ax4, eeg_c4_m1_unfiltered)
            plot_specta(ax5, eeg_f3_m2)
            plot_specta(ax6, eeg_f3_m2_unfiltered)
            plot_specta(ax7, eeg_f4_m1)
            plot_specta(ax8, eeg_f4_m1_unfiltered)
            plot_specta(ax9, eeg_o2_m1)
            plot_specta(ax10, eeg_o2_m1_unfiltered)
            plot_specta(ax11, eeg_o1_m2)
            plot_specta(ax12, eeg_o1_m2_unfiltered)
        def plot_avg_spindle():
            #fig = plt.gcf()
            #fig.size
            plt.rcParams.update({"font.size": 18})
            sp.plot_average(center="Peak", time_before=1, time_after=1)
            panel_filename = "_".join(panel.split("_")[:-1])
            #plt.title(f"Average spindle of recording: {panel_filename}")
        computed_recall, computed_precision, computed_f1 = compute_recall_precision_f1_metrics(all_ground_truth_spindles, all_yasa_detected_spindles, ovt=OVT)
        computed_TP, computed_FP, computed_FN = compute_confusion_metrics(all_ground_truth_spindles, all_yasa_detected_spindles, ovt=OVT)
        all_recalls[panel_name] = computed_recall
        all_precisions[panel_name] = computed_precision
        all_f1s[panel_name] = computed_f1
        all_TPs[panel_name] = computed_TP
        all_FPs[panel_name] = computed_FP
        all_FNs[panel_name] = computed_FN

        #evaluate_pairs(best_matching_pairs_global)
        if PLOT_MATRIX:
            plot_matrix_for_func(all_ground_truth_spindles, all_yasa_detected_spindles, overlap_metric)
        #plot_avg_spindle()
        if PLOT_PAIRS:
            plot_pairs()
        #plot_unfiltered_vs_filtered_eeg()
        #plot_confusion_matrix()
        plt.show()

    panel_keys = all_TPs.keys()

    print("All TPs: ")
    pprint(all_TPs)
    print("All FPs: ")
    pprint(all_FPs)
    print("All FNs: ")
    pprint(all_FNs)
    print("All recalls:")
    pprint(all_recalls)
    print("All precisions:")
    pprint(all_precisions)
    print("All F1s:")
    pprint(all_f1s)

    all_TPs_clean = {k:v for k, v in all_TPs.items() if v != "N/A"}
    all_FPs_clean = {k:v for k, v in all_FPs.items() if v != "N/A"}
    all_FNs_clean = {k:v for k, v in all_FNs.items() if v != "N/A"}
    all_recalls_clean = {k:v for k, v in all_recalls.items() if v != "N/A"}
    all_precisions_clean = {k:v for k, v in all_precisions.items() if v != "N/A"}
    all_f1s_clean = {k:v for k, v in all_f1s.items() if v != "N/A"}

    avg_TP = sum(all_TPs_clean.values()) / len(all_TPs_clean)
    avg_FP = sum(all_FPs_clean.values()) / len(all_FPs_clean)
    avg_FN = sum(all_FNs_clean.values()) / len(all_FNs_clean)

    avg_recall = sum(all_recalls_clean.values()) / len(all_recalls_clean)
    avg_precision = sum(all_precisions_clean.values()) / len(all_precisions_clean)
    avg_f1 = sum(all_f1s_clean.values()) / len(all_f1s_clean)

    print(f"Average TP: {avg_TP:.2f}")
    print(f"Average FP: {avg_FP:.2f}")
    print(f"Average FN: {avg_FN:.2f}")
    print(f"Average recall: {avg_recall:.2f}")
    print(f"Average precision: {avg_precision:.2f}")
    print(f"Average F1: {avg_f1:.2f}")

    def fmt_metric(metric):
        if metric == "N/A":
            return metric
        else:
            return round(metric, 2)

    for key in panel_keys:
        TP = all_TPs[key]
        FP = all_FPs[key]
        FN = all_FNs[key]
        recall = all_recalls[key]
        precision = all_precisions[key]
        F1 = all_f1s[key]
        panel_id = key.split("_")[-1]
        print(f"{panel_id} & {fmt_metric(TP)} & {fmt_metric(FP)} & {fmt_metric(FN)} & {fmt_metric(recall)} & {fmt_metric(precision)} & {fmt_metric(F1)} \\\\")
        print("\\hline")
    print("\\rowcolor{cyan!10}")
    print(f"Average & {fmt_metric(avg_TP)} & {fmt_metric(avg_FP)} & {fmt_metric(avg_FN)} & {fmt_metric(avg_recall)} & {fmt_metric(avg_precision)} & {fmt_metric(avg_f1)} \\\\")
    print("\\hline")





def overlap_metric(E, D):
    """Computes (E intersect D) / (E union D), ala A7 paper (Lacourse et al.)

    Args:
        E (startpoint (float), endpoint (float)): N/A
        D (startpoint (float), endpoint (float)): N/A

    Returns:
        overlap threshold (float): Overlap threshold
    """
    E_interesect_D = range_overlap(E, D)
    E_union_D = range_union(E, D)
    try:
        return E_interesect_D / E_union_D
    except ZeroDivisionError:
        logger.warning(f"encountered zero division in overlap_metric, E: {E}, D: {D}, E_intersects_D: {E_interesect_D}, E_union_D: {E_union_D}")
        return 0

def get_overlapping_spindles(real, preds):
    max_pairs, real_left, preds_left = compute_max_pairs(real, preds, overlap_metric)
    return max_pairs, real_left, preds_left


def compute_confusion_metrics(real, preds, ovt=0.2):
    """Computes TP, FP, FN

    Args:
        real ([type]): [description]
        preds ([type]): [description]
        ovt (float, optional): [description]. Defaults to 0.2.
    """
    max_pairs, real_left, preds_left = get_overlapping_spindles(real, preds)
    true_positives = [(score, r, p) for score, r, p in max_pairs if score >= ovt]
    false_positives = preds_left + [(r, p) for score, r, p in max_pairs if score < ovt]
    false_negatives = real_left

    TP = len(true_positives)
    FP = len(false_positives)
    FN = len(false_negatives)

    return (TP, FP, FN)

def compute_recall_precision_f1_metrics(real, preds, ovt=0.2):
    """Computes the recall (TP/(TP+FN)), precision (TP/(TP+FP)), and the F1-score (2*((Precision*recall)/(Precision+recall)))
    Ala 10.1016/j.jneumeth.2018.08.014 (DOI)

    Args:
        real ([type]): [description]
        preds ([type]): [description]
        ovt (float, [0,1]): Overlap threshold, how much overlap is needed between spindles to be considered identical 
    """
    TP, FP, FN = compute_confusion_metrics(real, preds, ovt)

    logger.info(f"TP: {TP}, FP: {FP}, FN: {FN}")

    try:
        recall = TP/(TP+FN)
    except ZeroDivisionError:
        logger.warning("recall computation caused ZeroDivision error")
        recall = "N/A"
    try:
        precision = TP/(TP+FP)
    except ZeroDivisionError:
        logger.warning("precision computation caused ZeroDivision error")
        precision = "N/A"
    # if neither recall or precision are None, i.e. both have values
    if "N/A" not in [recall, precision] and precision + recall > 0:
        F1 = 2*((precision*recall)/(precision+recall))
    else:
        F1 = "N/A"
    logger.info(f"len(real): {len(real)}, len(preds): {len(preds)}")
    if "N/A" not in [recall, precision] and precision + recall > 0:
        logger.info(f"recall: {recall:.2f}, precision: {precision:.2f}, F1: {F1:.2f}")
    else:
        logger.info(f"recall: {recall}, precision: {precision}, F1: {F1}")

    return (recall, precision, F1)

def plot_matrix_for_func(real, preds, func):
    """Given real ranges, and prediction ranges, and a function, plot a heatmap table of the results of applying func(r, p) for each r in real, p in preds

    Args:
        real (list of ranges): [description]
        preds (list of ranges): [description]
        func ([type]): [description]
    """
    overlap_mat = np.array([[func(r, p) for p in preds] for r in real])

    #plt.tick_params(top=True, bottom=False,
    #            labeltop=True, labelbottom=False)

    ax = sns.heatmap(overlap_mat, cbar_kws={'label': "Overlap metric ($O_{ED}$)"})
    ax.set_ylabel("Expert scored spindles")
    ax.set_xlabel("YASA detected spindles")



    # how many less times should the ticks appear, compared to default
    tick_reduction_factor = 2

    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % tick_reduction_factor != 0:
            label.set_visible(False)
    for index, label in enumerate(ax.yaxis.get_ticklabels()):
        if index % tick_reduction_factor != 0:
            label.set_visible(False)

    #spaced_xticks = [i*tick_reduction_factor for i in ax.get_xticks()][::tick_reduction_factor]
    #spaced_yticks = [i*tick_reduction_factor for i in ax.get_yticks()][::tick_reduction_factor]
    #ax.set_xticks(spaced_xticks)
    #ax.set_yticks(spaced_yticks)


    print(f"len(real): {len(real)}")
    print(f"len(preds): {len(preds)}")
    #ax.set_xticks(range(0, len()))

    # this forces matplotlib the rotation of the x and y-axis ticks to be "normal", i.e. facing downwards.
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)


    plt.show()






    

def plot():
    for panel in info:
        data_path = info[panel]["data_file"]
        data_file = h5py.File(data_path)
        scoring_path = info[panel]["scoring_file"]
        logger.debug("data file: %s" % data_path)
        logger.debug("scoring file: %s" % scoring_path)
        scores = pd.read_csv(scoring_path)
        scoring_seconds_ranges = info[panel]["scoring_seconds_ranges"]
        epoch_scores = []

        eeg_c3_m2 = Signal(data_file["C3-M2"][()], "C3-M2", data_file["C3-M2"].attrs["fs"])
        eeg_f3_m2 = Signal(data_file["F3-M2"][()], "F3-M2", data_file["F3-M2"].attrs["fs"])
        eeg_c4_m1 = Signal(data_file["C4-M1"][()], "C4-M1", data_file["C4-M1"].attrs["fs"])
        eeg_f4_m1 = Signal(data_file["F4-M1"][()], "F4-M1", data_file["F4-M1"].attrs["fs"])
        eeg_o2_m1 = Signal(data_file["O2-M1"][()], "O2-M1", data_file["O2-M1"].attrs["fs"])
        eeg_o1_m2 = Signal(data_file["O1-M2"][()], "O1-M2", data_file["O1-M2"].attrs["fs"])

        # convert to be of uv scale, yasa requires this
        eeg_c3_m2.data *= 1000000
        eeg_f3_m2.data *= 1000000
        eeg_c4_m1.data *= 1000000
        eeg_f4_m1.data *= 1000000
        eeg_o2_m1.data *= 1000000
        eeg_o1_m2.data *= 1000000


        seconds_count = np.linspace(start=0, stop=len(eeg_f4_m1)/200, num=len(eeg_f4_m1))
        plt.plot(seconds_count, eeg_c3_m2.data)
        for start, stop in zip(scores["Onset seconds"], scores["Onset seconds"] + scores["Duration"]):
            plt.axvspan(start, stop, alpha=0.5, color="red")
            logger.debug(f"area between {start} - {stop}")
        plt.show()



def count_expert_lengths():
    avgs = []
    sums = []
    for panel in info:
        data_path = info[panel]["data_file"]
        data_file = h5py.File(data_path)
        scoring_path = info[panel]["scoring_file"]
        #logger.debug("data file: %s" % data_path)
        #logger.debug("scoring file: %s" % scoring_path)
        #logger.info("Procesing panel: %s" % panel)
        scores = pd.read_csv(scoring_path)
        recording_start_period = convert_str_to_datetime(scores["Recording start"][0])
        scoring_seconds_ranges = info[panel]["scoring_seconds_ranges"]

        starts = scores["Onset seconds"]
        durations = scores["Duration"]
        stops = starts.add(durations)

        durations = stops.subtract(starts)
        print(f"Panel {panel}")
        print(f"Mean expert scored spindle length: {durations.mean():.2f}")
        print(f"Max expert scored spindle length: {durations.max():.2f}")
        print(f"Min expert scored spindle length: {durations.min():.2f}")
        print("----------------")
        #stops = scores["Start"]

        avgs.append(durations.mean())
        sums.append(durations.sum())

    print(f"Average over all panels: {sum(avgs)/len(avgs):.2f}")
    print(f"Sum over all panels: {sum(sums):.2f}")




if __name__ == "__main__":
    verify()
    #report_sleep_stages_for_scored_epochs()
    #count_expert_lengths()
    #plot()