import supereeg as se
import numpy as np
import pandas as pd
import ramutils
from ptsa.data.readers import EEGReader
from ramutils.events import load_events, clean_events, select_word_events
from ramutils.parameters import FilePaths, FRParameters, PS5Parameters
import os



start_time = 0.0
end_time = 1.6


sessions = np.unique(word_events['session'])
experiment = 'FR1'
fr_events = load_events(subject, experiment, sessions=sessions,rootdir=paths.root)


