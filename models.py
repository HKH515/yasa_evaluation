import math
import json
import datetime


def get_duration(obj):
    return datetime.timedelta(seconds=len(obj)/obj.sampling_frequency)

def get_seconds(obj):
    return get_duration(obj).total_seconds()

class Signal():
    def __init__(self, data, name, sampling_frequency):
        self.data = data
        self.name = name
        self.sampling_frequency = sampling_frequency
    
    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "%s (Signal):\n\tsampling_frequency: %s\n\tlength of data: %s\n\tduration: %s (%s seconds)" % (
            self.name,
            self.sampling_frequency,
            len(self),
            str(get_duration(self)),
            str(get_seconds(self))
        )
    def __repr__(self):
        return f"{self.name} (Signal)"
    
    def get_duration(self):
        return datetime.timedelta(seconds=len(self)/self.sampling_frequency)

class MultiSignal():
    def __init__(self, concat_data, names, sampling_frequency):
        """[summary]

        Args:
            concat_data (2d numpy array, vstacked eeg channels): [description]
            names (list of strings): [description]
            sampling_frequency ([type]): [description]
        """
        self.concat_data = concat_data
        self.names = names
        self.sampling_frequency = sampling_frequency
    
    def __len__(self):
        return len(self.concat_data.shape)

    def __str__(self):

        return "%s (MultiSignal):\n\tsampling_frequency: %s\n\shape of data: %s\n" % (
            ", ".join([name for name in self.names]),
            self.sampling_frequency,
            len(self)
        )
    def __repr__(self):
        return "%s (MultiSignal)" % ",".join([name for name in self.names])
    
    def get_duration(self):
        return datetime.timedelta(seconds=len(self)/self.sampling_frequency)

class AuxSignal():
    def __init__(self, name, bool_vector, sampling_frequency, ranges=None):
        self.name = name
        self.bool_vector = bool_vector
        self.sampling_frequency = sampling_frequency
        self.ranges = ranges
    
    def __len__(self):
        return len(self.bool_vector)

    def __str__(self):
        return f"-----------------------\nName: {self.name}\nLength of bool vector({len(self.bool_vector)})\nSampling frequency: {self.sampling_frequency}\nRanges: {self.ranges}\nDuration: {get_duration(self)} ({get_seconds(self)} seconds)\n-----------------------"
    def __repr__(self):
        return f"{self.name} (AuxSignal)"