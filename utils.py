from functools import reduce # Python version 3.x
from scipy.signal import hanning, welch
from scipy import fft
import numpy as np
from tqdm import tqdm
import math
import yasa
from datetime import datetime
from datetime import timedelta
import copy
from loguru import logger



import mne
from math import gcd # Python versions 3.5 and above

from models import Signal, AuxSignal, MultiSignal


import matplotlib.pyplot as plt

EPS = 0.00001

keys_to_use = ["Sigga", "Sigga ", "Sigga PESrescore"]


time_format = "%Y-%m-%dT%H:%M:%S.%f"
period_recording_format = "%Y/%m/%dT%H:%M:%S.%f"

# The two dicts below are defined as a workaround for a bug that arose when converting from .ndf files to .h5 files, namely, the .ndf files apparently don't store the sampling frequency of the original signal (e.g. for EEGs it would be 200hz). Using 200hz caused micro-drifts to appear later on in the signal, however using these sampling frequencies all is good. The values were derived by dividing the total number of samples in a recording, by the number seconds the recording spans.
eeg_sampling_frequencies = {
    "VSN_14_080_001": 200.013232646027,
    "VSN_14_080_003": 200.013292597358,
    "VSN_14_080_004": 200.013232497221,
    "VSN_14_080_005": 200.01321419551,
    "VSN_14_080_006": 200.013070334795,
    "VSN_14_080_007": 200.013009537456,
    "VSN_14_080_008": 200.013155361713,
    "VSN_14_080_009": 200.013154022773,
    "VSN_14_080_010": 200.01307969573,
    "VSN_14_080_011": 200.013204936936,
    "VSN_14_080_012": 200.013465942497,
    "VSN_14_080_013": 200.013094681976,
    "VSN_14_080_014": 200.013246009431,
    "VSN_14_080_015": 200.013294228864,
    "VSN_14_080_016": 200.013234801535,
    "VSN_14_080_017": 200.01322650487,
    "VSN_14_080_018": 200.013052511073,
    "VSN_14_080_019": 200.013060830514,
    "VSN_14_080_020": 200.013116929742,
    "VSN_14_080_021": 200.013109118017,
    "VSN_14_080_022": 200.013135399598,
    "VSN_14_080_023": 200.013050572027,
    "VSN_14_080_024": 200.013098799188,
    "VSN_14_080_025": 200.013172377786,
    "VSN_14_080_026": 200.01306741491,
    "VSN_14_080_027": 200.013034964173,
    "VSN_14_080_028": 200.013118101722,
    "VSN_14_080_029": 200.013202189014,
}

spo2_sampling_frequencies = {
    "VSN_14_080_001": 2.99994683482563,
    "VSN_14_080_003": None,
    "VSN_14_080_004": 2.99997440581438,
    "VSN_14_080_005": 3.00003463021851,
    "VSN_14_080_006": 2.99998277666239,
    "VSN_14_080_007": 3.00002044345853,
    "VSN_14_080_008": 2.99996164129074,
    "VSN_14_080_009": 3.00001806241105,
    "VSN_14_080_010": 2.99997822203635,
    "VSN_14_080_011": 2.99994649303067,
    "VSN_14_080_012": 3.00007620878581,
    "VSN_14_080_013": 3.00001739741307,
    "VSN_14_080_014": 3.00001855404331,
    "VSN_14_080_015": 3.00005873294439,
    "VSN_14_080_016": 2.99998543000502,
    "VSN_14_080_017": 3.00014887833327,
    "VSN_14_080_018": 2.99997977163389,
    "VSN_14_080_019": 2.99997949917328,
    "VSN_14_080_020": 2.99992629779552,
    "VSN_14_080_021": 2.99994552150471,
    "VSN_14_080_022": 2.99999454106377,
    "VSN_14_080_023": 2.99992868519435,
    "VSN_14_080_024": 3.00007468271471,
    "VSN_14_080_025": 3.00000640309208,
    "VSN_14_080_026": 3.00009435375315,
    "VSN_14_080_027": None,
    "VSN_14_080_028": 3.00013697137418,
    "VSN_14_080_029": 2.99988603325064,
}

class ScoringKeyException(Exception):
    pass


def resamplify_onsets(scores, event_name, scored_hz, target_hz=200):
    """The onsets that are exported from Noxturnal are most likely not correctly aligned with the data being annotated, due to the sampling frequency being computed on the fly (dividing number of samples by the number of seconds a recording is)

    formula we apply to the onsets: (onset * scored_hz) / target_hz
    onset * scored_hz is effectively the onset but measured in samples (how far from the beginning does this event occur at, in samples), this is then divided by the target frequency, to get the correct placement in time.

    Args:
        scores (output of json.read): Contents of json.read on scoring.json of a panel
        event_name (string): Event name to apply the resampling to, e.g. "oxygendesaturation-drop"
        scored_hz (float): Sampling frequency as reported by the NDF file (typically around 200.0132)
        target_hz (float, optional): Sampling frequency of the signal being scored. Defaults to 200.
    """

    scores_copy = copy.deepcopy(scores)

    for key in scores.keys():
        for event_index, event in enumerate(scores[key]):
            if isinstance(event, dict):
                if scores[key][event_index]["label"] == event_name:
                        scores_copy[key][event_index]["onset"] *= scored_hz
                        scores_copy[key][event_index]["onset"] /= target_hz
    return scores_copy

def resamplify_ranges(ranges, scored_hz, target_hz=200):
    """The onsets that are exported from Noxturnal are most likely not correctly aligned with the data being annotated, due to the sampling frequency being computed on the fly (dividing number of samples by the number of seconds a recording is)

    formula we apply to the onsets: (onset * scored_hz) / target_hz
    onset * scored_hz is effectively the onset but measured in samples (how far from the beginning does this event occur at, in samples), this is then divided by the target frequency, to get the correct placement in time.

    Args:
        scores (list of tuples): Each tuple denotes a start and stop point in time, from the beginning of recording. Ex: scores = [(15.63, 15.916), (204.152, 204.61)]
        scored_hz (float): Sampling frequency as reported by the NDF file (typically around 200.0132)
        target_hz (float, optional): Sampling frequency of the signal being scored. Defaults to 200.
    """

    ranges_copy = []

    for start, stop in ranges:
        corrected_start = (start * scored_hz) / target_hz
        corrected_stop = (stop * scored_hz) / target_hz

        ranges_copy.append((corrected_start, corrected_stop))
    return ranges_copy

def resamplify_dataframe(df, scored_hz, target_hz=200):
    """The onsets that are exported from Noxturnal are most likely not correctly aligned with the data being annotated, due to the sampling frequency being computed on the fly (dividing number of samples by the number of seconds a recording is)

    formula we apply to the onsets: (onset * scored_hz) / target_hz
    onset * scored_hz is effectively the onset but measured in samples (how far from the beginning does this event occur at, in samples), this is then divided by the target frequency, to get the correct placement in time.

    Args:
        df (pandas.DataFrame): DataFrame describing the spindle scorings.
        scored_hz (float): Sampling frequency as reported by the NDF file (typically around 200.0132)
        target_hz (float, optional): Sampling frequency of the signal being scored. Defaults to 200.
    """



    df["Onset seconds"] *= scored_hz
    df["Onset seconds"] /= target_hz
    return df


def compute_max_pairs(list1, list2, func):
    """Given two lists, for each element x in list1 finds the element y in list2 such that func(x, y) is maximal, this is achieved by creating a list of tuples, each tuple of the form (func(list1_element, list2_element), list1_element, list2_element), and sorting in descending order, this will sort it such that the maximal pairs occur first.

    Args:
        list1 (list): N/A
        list2 (list): N/A
        func (function that takes in two parameters and returns a value): Function that you want to maximize
    Returns:
        opt (list of tuples (score, list1_element, list2_element)): List of optimal pairings, maximized by func
        list1_left: Elements from list1 that did not get paired
        list2_left: Elements from list2 that did not get paired

    """

    # this will be filled with tuples of the form(func(x, y), x , y)
    triples = []
    for x in list1:
        for y in list2:
            triples.append((func(x, y), x, y))
    # sort in descending order
    triples = sorted(triples, reverse=True)

    list1_left = copy.deepcopy(list1)
    list2_left = copy.deepcopy(list2)

    # list will be filled with tuples of (x, y), optimal pairs found
    opt = []

    for score, x, y in triples:
        if x in list1_left and y in list2_left and score > 0.0:
            opt.append((score, x, y))
            try:
                list1_left.remove(x)
            except ValueError:
                print("error in compute_max_pairs when removing from list")
            try:
                list2_left.remove(y)
            except ValueError:
                print("error in compute_max_pairs when removing from list")
    return (opt, list1_left, list2_left)
            
def range_union(range1, range2):
    """Returns the size of the union of two ranges.

    Example:
        range1: |-------|
        range2:   |---------|
        output: |-----------|

    Args:
        range1 ([type]): [description]
        range2 ([type]): [description]
    """
    return max(0, max(range1[1], range2[1]) - min(range1[0], range2[0]))

def range_overlap(range1, range2):
    """Returns the size of the intersection of two ranges by subtracting the leftmost endpoint (of either range) from the rightmost startpoint (of either range)
    
    Example:
        range1: |-------|
        range2:   |---------|
        output:   |-----|


    Args:
        range1 (startpoint (float), endpoint (float)): N/A
        range2 (startpoint (float), endpoint (float)): N/A

    Returns:
        overlap (float): Size of the overlap
    """
    return max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))

def range_overlap_coefficient(range1, range2):
    """Computes https://en.wikipedia.org/wiki/Overlap_coefficient

    Args:
        range1 (startpoint (float), endpoint (float)): N/A
        range2 (startpoint (float), endpoint (float)): N/A

    Returns:
        overlap coefficient (float): Overlap coefficient
    """
    overlap = range_overlap(range1, range2)
    smaller_range = min([range1, range2], key=lambda x: max(x) - min(x))
    smaller_range_size = max(smaller_range) - min(smaller_range)
    return overlap / smaller_range_size


def convert_str_to_datetime(time_string, formatting=time_format):
    return datetime.strptime(time_string, formatting)

def compute_desat_indices(arr):
    """
    doesn't really work, does not give the results that human scorers give, understandably.
    This algorithm is very conservative when saying something is a desat.
    """
    lookahead = 1000
    lookahead_spread = 100
    desat_indices = []

    for i in tqdm(range(lookahead, len(arr))):
        curr = arr[i]
        # reading error
        if curr == 0.0:
            continue
        for l in range(1, lookahead, lookahead_spread):
            comparison = arr[i-l]
            # reading error
            if comparison == 0.0:
                continue
            change_percent = abs(curr - comparison) / ((curr + comparison) / 2)
            if change_percent > 0.03 and comparison > curr:
                logger.debug("change of %s percent (curr: %s, comparison: %s)" % (change_percent * 100, curr, comparison))
                desat_indices.extend(list(range(i-l, i)))

    desat_indices = sorted(list(set(desat_indices)))
    logger.debug(desat_indices)

    return desat_indices

# given EEG data, extracts the delta band information
def extract_delta_band(arr, sampling_freq, ch_names):
    window = hanning(len(arr))
    #windowed_arr = arr * window
    N = len(arr)//(2+1)
    X = np.linspace(0, sampling_freq, N, endpoint=True)
    spect = fft(arr)

    bandpowers = yasa.bandpower(arr, sf=sampling_freq, ch_names=ch_names)
    logger.info(bandpowers)

# given EEG data, extracts the delta band information
def extract_delta_band_mne(arr, sampling_freq):

    arr_2d = np.expand_dims(arr, axis=0)
    info = mne.create_info(ch_names=["EEG C3-M2"], sfreq=sampling_freq, ch_types=["eeg"])
    raw = mne.io.RawArray(arr_2d, info)

    scalings = {"eeg" : 2}
    raw.plot(n_channels=1, scalings=scalings, show=True, block=True)

# taken from https://raphaelvallat.com/bandpower.html
def plot_welch(ax, arr, sampling_rate):
    # Define window length (4 seconds)
    win = 4 * sampling_rate
    freqs, psd = welch(arr, sampling_rate, nperseg=win)

    # Plot the power spectrum
    #sns.set(font_scale=1.2, style='white')
    #plt.figure(figsize=(8, 4))
    ax.plot(freqs, psd, color='k', lw=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density (V^2 / Hz)', fontsize=6)
    ax.set_ylim([0, psd.max() * 1.1])
    ax.set_title("Welch's periodogram")
    ax.set_xlim([0, freqs.max()])
    #sns.despine()
    #plt.show()


def get_key_from_scoring(scores):
    for key in keys_to_use:
        if key in scores:
            return key


def determine_scoring_key(scores, desired_fields):
    """
    scores: should be the result of json.load()
    desired_fields: list of names of signals you would like
    """

    logger.debug("trying to find suitable key from the following keys: %s" % ", ".join(scores.keys()))

    for key in scores:
        if type(scores[key]) != list:
            # the current scoring key value is not an array of recordings
            continue
        # each entry in bitmap corresponds to whether a desired_field occurs as the value of "label" somewhere in the array
        bitmap = [x in [y["label"] for y in scores[key]] for x in desired_fields]
        if all(bitmap):
            return key
    raise ScoringKeyException("No suitable scoring key found for fields: %s" % (", ".join(desired_fields)))

def find_starttime_based_on_events(events):
    """Given a list of events, finds the earliest timestamp entry for 'Start' and returns that

    Args:
        events (list of dicts): List of events, i.e. an entry in the scores object (scores[key])
    """
    earliest_event = events[0]

    for event in events:
        if convert_str_to_datetime(event["start"]) < convert_str_to_datetime(earliest_event["start"]):
            earliest_event = event


    return earliest_event["start"]

def read_scored_desat_indices(scores):
    """
    it's crucial that scores has been processed by get_range_between_first_epoch_and_last_epoch_onset()
    """
    logger.debug(scores.keys())

    # since the names of the keys (sigga, psg-marta, sigga/marta) can be inconsistent, we want to search for a good key to use
    #key_to_use = determine_scoring_key(scores, ["oxygensaturation-drop"])
    # 19.11.2020: determine_scoring_key was giving us Untitled-1, occasionally, this is an automatic scoring from noxturnal, we'd rather stick with Siggas scorings
    key_to_use = get_key_from_scoring(scores)

    logger.debug("using key %s" % key_to_use)
    desat_ranges = []

    #first_start_dt = datetime.strptime(find_starttime_based_on_events(scores[key_to_use]), time_format)
    # 05.02.2021: this will load the data in starting from a different point than other scores and signals are, need to fix to start from the first epoch within the period analysis
    #period_recording = get_period_recording_from_scores(scores)
    #first_start_dt = convert_str_to_datetime(scores["Recording start"][0])

    for event in scores[key_to_use]:
        if event["label"] == "oxygensaturation-drop":
            desat_ranges.append((event["onset"], event["onset"] + event["duration"]))

    logger.debug("Desat ranges: ")
    logger.debug(desat_ranges)
    return desat_ranges

def get_period_recording_from_scores(scores):
    return scores["Recording"]["start"]

# This is an attempt to rewrite the function such that it outputs a boolean vector of the correct size, i.e. the length of the eeg or the LCM frequency or something like that 
def read_scored_desat_indices_as_bool_vector(scores, sampling_rate, upsampled_length):
    """
    Returns a bool vector, listing for each sample (i.e. each data point recorded, e.g. for EEG) whether that datapoint is inside a desaturated event marked in scores
    The reason why upsampled_length is used, is because we initially computed the last index by doing max(scores) * sampling_rate, but that gave us incorrect results.
    """
    desats = read_scored_desat_indices(scores)
         
    # we floor after multiplying with sampling rate so the rounding error is much less
    rescaled_desats = [(math.floor(beg*sampling_rate), math.floor(end*sampling_rate)) for beg, end in desats]
    assert(all([type(beg) == int and type(end) == int for beg, end in rescaled_desats]))

    #max_sample = max([max(a, b) for a, b in rescaled_desats])
    logger.debug("upsampled_length: %s" % upsampled_length)
    #logger.debug("max_sample: %s" % max_sample)


    bool_vector = [0 for i in range(upsampled_length)]
    for start, end in rescaled_desats:
        # if the desat that we're looking at exceeds the EEG length, we skip it, this is due to an issue where some events (including desats) are scored after the eeg ends, this may be due to the nox reader api, or it might be due to exporting script, I am not sure, for now we just skip those ranges.
        if start > upsampled_length or end > upsampled_length:
            break
        for i in range(int(start), int(end)):
            bool_vector[i] = 1

    return np.array(bool_vector)


def detect_spindles(eeg_Signal, multi_only=False, remove_outliers=False, hypno=None, sleep_stages_to_include=(1, 2, 3)):
    """
    eeg_Signal can either be a Signal or a MultiSignal

    uses yasa (yet another spindle algorithm (https://github.com/raphaelvallat/yasa)) to find sleep spindles.
    code is based on https://github.com/raphaelvallat/yasa/blob/master/notebooks/01_spindles_detection.ipynb

    multi_only refers to "Should I only pick up spindles that span across 2 or more EEG channels?"
    returns bool array
    """
    if type(eeg_Signal) == Signal:
        # force multi_only, as this is only for 1 channel
        sp = yasa.spindles_detect(eeg_Signal.data, eeg_Signal.sampling_frequency, multi_only=False, hypno=hypno, remove_outliers=remove_outliers, include=sleep_stages_to_include)
        logger.debug(type(sp))
        bool_spindles = sp.get_mask()
        return bool_spindles
    elif type(eeg_Signal) == MultiSignal:
        sp = yasa.spindles_detect(eeg_Signal.concat_data, eeg_Signal.sampling_frequency, ch_names=eeg_Signal.names, multi_only=multi_only, hypno=hypno, remove_outliers=remove_outliers, include=sleep_stages_to_include)
        logger.debug(type(sp))
        bool_spindles = sp.get_mask()
        # apparently YASA returns a 2d numpy arr when you put in 2d eeg data (channels, samples), so we want to apply bit operation OR on each row, to just pick up, using technique found here: https://stackoverflow.com/questions/44671407/how-to-perform-row-wise-or-operation-on-a-2d-numpy-array
        # 0 is the row axis, i.e. we are "collapsing the rows"
        bool_spindles = np.bitwise_or.reduce(bool_spindles, 0)
        logger.debug(f"shape of bool_spindles mask: {bool_spindles.shape}")
        return bool_spindles
    else:
        raise TypeError(f"Illegal argument type for eeg_Signal: {type(eeg_Signal)}")

def plot_spindles(ax, channel_name, data, data_sampling_freq, bool_spindles):
    spindles_highlight = data * bool_spindles
    spindles_highlight[spindles_highlight == 0] = np.nan
    times = np.arange(len(data)) / data_sampling_freq

    ax.set_title("Channel '%s', green is sleep spindles" % channel_name)
    #plt.figure(figsize=(14, 4))
    ax.plot(times, data)
    ax.plot(times, spindles_highlight, 'green')

    ax.set_ylabel("uV")

    #plt.xlim([0, times[-1]])

    #sns.despine()

def compute_overlap(aux1, aux2):
    """Computes the % of samples that are present in aux signal A and aux signal B

    Args:
        a (AuxSignal): N/A
        b (AuxSignal): N/A
    """
    assert len(aux1) == len(aux2)
    shared_samples = 0
    for i in range(len(aux1)):
        if aux1.bool_vector[i] == aux2.bool_vector[i]:
            shared_samples += 1
    return shared_samples / len(aux1)
    

def float_equals(val1, val2, eps=0.0001):
    """this function is used when comparing floats to avoiding issues relating to floating point imprecision (i.e. 20 might be evaluated as 20.00000001, thus comparing that value to 20 will result in false, but using this function it should be fine with the right epsilon)

    Args:
        val1 ([type]): [description]
        val2 ([type]): [description]
        eps ([type], optional): [description]. Defaults to -1e04.

    Returns:
        [type]: [description]
    """
    return abs(val1 - val2) < eps

def apply_noxturnal_filtering(signal):
    signal_copy = Signal(signal.data, signal.name, signal.sampling_frequency)
    signal_copy.data = np.ndarray.astype(signal_copy.data, np.float64)
    filtered_signal_data = mne.filter.notch_filter(signal_copy.data, signal_copy.sampling_frequency, freqs=[50])
    filtered_signal_data = mne.filter.filter_data(filtered_signal_data, signal_copy.sampling_frequency, l_freq=0.3, h_freq=35)
    return Signal(name=signal_copy.name, data=filtered_signal_data, sampling_frequency=signal_copy.sampling_frequency)

def apply_powerline_filtering_only(signal, verbose=True):
    signal_copy = Signal(signal.data, signal.name, signal.sampling_frequency)
    signal_copy.data = np.ndarray.astype(signal_copy.data, np.float64)
    filtered_signal_data = mne.filter.notch_filter(signal_copy.data, signal_copy.sampling_frequency, freqs=[50], verbose=verbose)
    return Signal(name=signal_copy.name, data=filtered_signal_data, sampling_frequency=signal_copy.sampling_frequency)
   
def get_period_analysis_range(scores):
    #key_to_use = get_key_from_scoring(scores)
    minval = float('inf')
    minarg = None
    minkey = None
    #minkey = list(scores.keys())[1]
    #minkey = get_key_from_scoring(scores)
    #minarg = (scores[minkey]["onset"], scores[minkey]["onset"] + scores[minkey]["duration"])

    for key in scores.keys():
        for event in scores[key]:
            # the first event can be a string, scoring json is weird like that
            if isinstance(event, dict):
                #print(event["label"])
                # if the period_analysis event we find is between 0 and 12 hours of sleep, and is not at start
                if event["label"] == "period_analysis" and event["duration"] < 43200:
                    if abs(event["onset"] - (event["onset"] + event["duration"])) < minval and event["duration"] > 0:
                        minarg = (event["onset"], event["onset"] + event["duration"])
                        minval = abs(event["onset"] - (event["onset"] + event["duration"]))
                        minkey = key
    #logger.debug(f"using period analysis key '{minkey}'")
    return minarg

def get_range_between_first_epoch_and_last_epoch_onsets(scores):
    """Returns the range from the first epoch that has been scored for a sleep stage, to the beginning of the last epoch
    This only includes epochs that are between the start and the end of the "period analysis".

    Args:
        scores ([type]): [description]

    Returns:
        [type]: [description]
    """

    period_analysis_range = get_period_analysis_range(scores)
    first_sleep_stage_onset = get_first_sleep_score_onset(scores)
    last_sleep_stage_onset = get_last_sleep_score_onset(scores)
    assert first_sleep_stage_onset > period_analysis_range[0] and last_sleep_stage_onset < period_analysis_range[1]
    return (first_sleep_stage_onset, last_sleep_stage_onset)


def get_range_between_first_epoch_and_last_epoch_onsets_over_scores(scores):
    range_start, range_end = get_range_between_first_epoch_and_last_epoch_onsets(scores)
    scores_copy = copy.deepcopy(scores)

    # we want to make sure we are only reading onset and duration from this modified scores object, otherwise we want an error
    fields_to_retain = ["onset", "duration", "label", "Start Time"]

    for key in scores.keys():
        deleted_items = 0
        for event_index, event in enumerate(scores[key]):
            assert event_index >= deleted_items
            if isinstance(event, dict):
                event_start = scores_copy[key][event_index-deleted_items]["onset"]
                event_end = event_start + scores_copy[key][event_index-deleted_items]["duration"]
                if event["label"] == "period_analysis":
                    # we do this, to ensure we arent reading from period_analysis of the modified json
                    del scores_copy[key][event_index-deleted_items]
                    deleted_items += 1
                #if event does not occur within range_start and range_end, remove
                elif event_start < range_start or event_end > range_end:
                    del scores_copy[key][event_index-deleted_items]
                    deleted_items += 1
                else:
                    available_keys = scores_copy[key][event_index-deleted_items].keys()
                    if "onset" in available_keys:
                        scores_copy[key][event_index-deleted_items]["onset"] -= range_start
                    #if "start" in available_keys:
                    #    start_of_event = float(scores_copy[key][event_index-deleted_items]["start"])
                    #    start_of_event -= range_start
                    #    scores_copy[key][event_index-deleted_items]["start"] = start_of_event
                    for field in event:
                        if field not in fields_to_retain:
                            del scores_copy[key][event_index-deleted_items][field]
    return scores_copy

def get_period_analysis_range_over_signal(signal, scores):
    period_analysis_range = get_period_analysis_range(scores)
    start_sample = int(period_analysis_range[0]*signal.sampling_frequency)
    end_sample = int(period_analysis_range[1]*signal.sampling_frequency)
    #print(f"length of signal: {len(signal)}")
    #print(f"start_sample: {start_sample}")
#print(f"end_sample: {end_sample}")
    return Signal(name=signal.name, data=signal.data[start_sample:end_sample], sampling_frequency=signal.sampling_frequency)

def get_range_between_first_epoch_and_last_epoch_onsets_over_signal(signal, scores):
    range_start, range_end = get_range_between_first_epoch_and_last_epoch_onsets(scores)
    start_sample = int(range_start * signal.sampling_frequency)
    end_sample = int(range_end * signal.sampling_frequency)

    return Signal(name=signal.name, data=signal.data[start_sample:end_sample], sampling_frequency=signal.sampling_frequency)



# order matters (because of YASA), wake is 0, n1 is 1, n2 is 2, n3 is 3, rem is 4
SLEEP_LABELS = ["sleep-wake", "sleep-n1", "sleep-n2", "sleep-n3", "sleep-rem"]


def get_first_sleep_score_onset(scores):
    key_to_use = get_key_from_scoring(scores)
    period_start, period_end = get_period_analysis_range(scores)
    
    for event in scores[key_to_use]:
        if event["label"] in SLEEP_LABELS and event["onset"] > period_start and event["onset"] <= (period_end-30):
            return event["onset"]
def get_last_sleep_score_onset(scores):
    key_to_use = get_key_from_scoring(scores)
    period_start, period_end = get_period_analysis_range(scores)
    for event in scores[key_to_use][::-1]:
        if event["label"] in SLEEP_LABELS and event["onset"] > period_start and event["onset"] <= (period_end-30):
            return event["onset"]

def get_hypno_list_from_scores(scores):
    #sleep_labels = ["sleep-wake", "sleep-n1", "sleep-n2", "sleep-n3", "sleep-rem"]
    # YASA only needs N1, N2, N3; also some panels (11 and 29) were being skipped because they didnt contain any scorings that contained rem sleep
    #key_to_use = determine_scoring_key(scores, sleep_labels)
    key_to_use = get_key_from_scoring(scores)

    start, end = get_range_between_first_epoch_and_last_epoch_onsets(scores)


    # hypno_list is a list of sleep stages for each epoch (currently hardcoded for 30 second epochs)
    hypno_list = []
    # ranges is a list of the format [(0, a, b), (1, c, d)] where there is a sleep stage 0 from a to b, 1 from c to d, etc.
    ranges = []
    current_time = start
    for event in scores[key_to_use]:
        if float_equals(current_time, end):
            break
        #if event["Event_Type"] in sleep_labels:
        #    print("%s, %s" % (event["Onset"], current_time + 30))
        # if the event we're looking at occurs exactly 30 seconds after the previous sleep score, and happens to be a sleep score
        if float_equals(event["onset"], (current_time + 30)) and event["label"] in SLEEP_LABELS:
            current_time += 30
            hypno_list.append(SLEEP_LABELS.index(event["label"]))
            ranges.append((SLEEP_LABELS.index(event["label"]), current_time-start-30, current_time-start))
        # if the event we're looking at is after 30 seconds, then we didn't get a sleep score this epoch, add artifact/movement label to mark it as invalid
        elif event["onset"] > (current_time + 30 + EPS):
            if event["label"] in SLEEP_LABELS:
                logger.warning("MARKING EVENT %s AS INVALID DUE TO TIME DIFFERENCE: %s + 30 != %s" % (event["label"], current_time-start, event["onset"]))
            current_time += 30
            hypno_list.append(-1)

    return (ranges, hypno_list)

def plot_specta(ax, signal):
    ax.specgram(signal.data, Fs=signal.sampling_frequency)
    ax.set_title(f"Spectagram ({signal.name})")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time (s)")


def plot_stages(ax, hypno_ranges):
    """
    hypno_ranges is a list of tuples of the form (s, a, b), indicating between a and b seconds in, s is the sleep stage assigned to that range.
    """
    colors = ["yellow", "orange", "red", "maroon", "purple"]
    for stage, start, end in hypno_ranges:
        ax.axvspan(start, end, color=colors[stage])
    #ax.legend(["w", "n1", "n2", "n3", "rem"])
        


def plot_o2(ax, arr, sampling_freq):
    times = np.arange(len(arr)) / sampling_freq
    ax.set_title("SPO2")
    ax.set_ylabel("o2 % sat.")
    ax.plot(times, arr)

def get_desat_aux_signal_from_scores(panel, scores, sampling_rate, length):
    """Given a json score object and a channel name, returns an aux signal object with that boolean vector loaded

    Args:
        panel (string): name of the panel, from where this aux signal is derived from
        scores ([type]): [description]
        channel_name ([type]): [description]
    """
    bool_vec = read_scored_desat_indices_as_bool_vector(scores, sampling_rate, length)
    return AuxSignal(f"{panel} - oxygensaturation-drop", bool_vec, sampling_rate)

def get_spindles_aux_signal_from_eeg_signal(panel, eeg_Signal, multi_only=False, remove_outliers=False, hypno=None):
    """Reads in an EEG signal and outputs a aux signal containing sleep spindles found in the eeg using YASA

    Args:
        panel (string): name of the panel, from where this aux signal is derived from
        eeg_Signal (Signal): Signal object containing EEG data

    Returns:
        AuxSignal: Sleep spindles
    """
    bool_vec = detect_spindles(eeg_Signal, remove_outliers=remove_outliers, hypno=hypno)
    return AuxSignal(f"{panel} - yasa-spindle", bool_vec, eeg_Signal.sampling_frequency)

def get_spindles_aux_signal_from_concat_eeg_signals(panel, eeg_MultiSignal, multi_only=False, remove_outliers=False, hypno=None):
    """Reads in a MultiSignal (bunch of EEGs stored together) and outputs a aux signal containing sleep spindles found in the eeg using YASA

    Args:
        panel (string): name of the panel, from where this aux signal is derived from
        eeg_MultiSignal (MultiSignal): Signal object containing EEG data

    Returns:
        AuxSignal: Sleep spindles
    """
    bool_vec = detect_spindles(eeg_MultiSignal, multi_only=multi_only, remove_outliers=remove_outliers, hypno=hypno)
    return AuxSignal(f"{panel} - yasa-spindle", bool_vec, eeg_MultiSignal.sampling_frequency)

def combine_spindles_across_channels(sp, min_distance=500, to_list=False):
    """Combines spindles that are less than min_distance milliseconds between each other, regardless of what channel they're on. This is very similar to how yasa.spindles_detect behaves, but yasa.spindles_detect does not combine across channels.
    
    Args:
        sp (yasa.SpindleResults): N/A
        min_distance: Milliseconds required between spindles to be considered distinct
    """

    summary = sp.summary()[["Start", "End"]]
    summary = summary.sort_values(by=["Start", "End"], axis=0)
    #logger.debug(summary)
    min_distance_s = min_distance / 1000
    # taken from https://stackoverflow.com/questions/48825078/efficient-way-of-merging-ranges-intervals-within-a-given-threshold
    combined_summary = (summary.groupby(summary['Start'].sub(summary['End'].shift()).ge(min_distance_s).cumsum()).agg({'Start':'first','End':'last'}))
    combined_summary_list = combined_summary.to_records(index=False).tolist()
    if to_list:
        return combined_summary_list
    else:
        return combined_summary

def combine_spindles_across_channels_yasa_style(sp, min_distance=500, sf=200):
    # Grab spindles
    summary = sp.summary()[["Start", "End"]]
    ranges = sorted(summary.to_records(index=False).tolist())
    #logger.debug(f"ranges: {ranges}")

    # Multiply values to be indices
    scaled_ranges = [(x*sf, y*sf) for x, y in ranges]
    #logger.debug(f"scaled_ranges: {scaled_ranges}")


    # Convert to numpy
    numpy_scaled_ranges = np.array(scaled_ranges)
    #logger.debug(f"numpy_scaled_ranges: {numpy_scaled_ranges}")

    # Send to yasa functions
    merged_events = yasa.others._merge_close(numpy_scaled_ranges, min_distance, sf)
    #logger.debug(f"merged_events: {merged_events}")

    # Convert back to seconds
    descaled_ranges = sorted([(x/sf, y/sf) for x, y in merged_events])
    #logger.debug(f"descaled_ranges: {descaled_ranges}")


def get_duration_of_start_stop_range(start, stop):
    delta = timedelta(seconds=stop-start)
    return delta