import json
import pandas as pd
import sys
import os
sys.path.append('/Users/tungphan/PycharmProjects/autoencoder_superEEG/autoencoder_single/cmlreaders/')
import cmlreader

rhino_root = "/Volumes/RHINO/"

# Instantiate the finder object
finder = cml.PathFinder(subject="R1389J", experiment="catFR5", session=1,
                        localization=0, montage=0, rootdir=rhino_root)

example_data_types = ['pairs', 'task_events', 'voxel_coordinates']
for data_type in example_data_types:
    print(finder.find(data_type))


from cmlreaders import get_data_index
r1_data = get_data_index(kind='r1', rootdir='/Volumes/RHINO/')
r1_data.head()
fr1_subjects = r1_data[r1_data['experiment'] == 'FR1']['subject'].unique()
fr1_subjects




reader = cml.CMLReader(subject="R1389J", experiment="catFR5", session=1,
                       localization=0, montage=0, rootdir=rhino_root)

