# yasa_evaluation
A tool that runs Yet Another Spindle Algorithm (YASA) on a proprietary dataset (VSN-14-080) using a proprietary scoring for spindles. This is the work behind my Masters thesis (link here: LINK).

## Installation instructions

1. You need to install the header files for hdf5, this is different between operating systems, further instructions can be found at https://docs.h5py.org/en/stable/build.html. However, for me (on Ubuntu 20.04) what worked was running `sudo apt-get install libhdf5-dev` and `sudo apt-get install libhdf5-serial-dev`.
2. Copy the repo by running `git clone https://github.com/hkh515/yasa_evaluation`
3. Enter the directory by running `cd yasa_evaluation`
3. Create a virtual env, by running `virtualenv venv --python=python3`, for reference, python 3.6.9 was used for the duration of the development cycle of this thesis.
4. Activate the venv by running `source venv/bin/activate`
5. Install all the packages by running `pip install -r requirements.txt`

## Usage


```bash
usage: spindle_verification.py [-h] [--verbose] [--multi_channel] [--plot_matrix] [--plot_pairs] [--ovt OVT] [--n2_only]
               [--all_but_n2] [--all_stages] [--channel {C,F}] [--remove_outliers] [--multi_only]

```
## Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
|`-v`|`--verbose`|`0`|`None`|
||`--multi_channel`||Use all EEG channels for spindle detection|
||`--plot_matrix`||Plot a heatmap of real vs predicted pairs|
||`--plot_pairs`||Plot a comparison between yasa and expert|
||`--ovt`|`0.2`|Overlap threshold, between 0.0 and 1.0 (both inclusive)|
||`--n2_only`||Validation will only involve looking at epochs which have been scored as N2 (default is N1, N2, and N3)|
||`--all_but_n2`||Validation will only involve looking at epochs which have not been scored as N2 (default is N1, N2, and N3)|
||`--all_stages`||Validation will involve looking at epochs regardless of sleep stage (default is N1, N2, and N3)|
||`--channel`|`None`|Determines the channel that YASA will analyze (if --multi_channel mode is not activated), also determines what expert scored spindles we will compare to (regardless of whether --multi_channel is on). Choices are 'C' and 'F'|
||`--remove_outliers`||Use an isolation forest algorithm to reject outlying spindles|
||`--multi_only`||Determines whether a spindle needs to be detected on 2 or more channels to be considered|
