import ptsa
import supereeg as se
import numpy as np
from ptsa.data.TimeSeriesX import TimeSeriesX
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import (
    MonopolarToBipolarMapper,
    MorletWaveletFilterCpp
)
from ptsa.data.readers.ParamsReader import ParamsReader
from collections import OrderedDict
import os
import glob
import time
from sklearn.externals import joblib
from classifier_randomized import*
import sys
#from helper_funcs import*
import h5py
args = sys.argv
index = int(args[1])


rhino_root = '/Volumes/RHINO'
dir = rhino_root + '/scratch/tphan/superautoencoder/'
subjects = np.sort(os.listdir(dir))


#index = 4
subject = subjects[index]
print(subject)

subject_dir = dir + subject + '/power.hdf5'

f = h5py.File(subject_dir, 'r')

orig_dataset = f['orig'].value
supereeg_dataset = f['recon'].value

dataset_list = []
for i,experiment in enumerate(['FR1', 'CatFR1']):
        try:
            dataset_orig = collections.OrderedDict()
            task_events_file = rhino_root + '/scratch/lucy.owen/supereeg/events/' + subject + '_RAM_' + experiment +  '/events.npy'
            task_events = np.load(task_events_file)
            dataset_orig['X'] = supereeg_dataset
            dataset_orig['y'] = task_events['recalled']
            dataset_orig['session'] = task_events['session']
            if experiment == 'CatFR1':
                dataset_orig['session'] = dataset_orig['session'] + 100
            dataset_orig['list'] = task_events['list']
            dataset_list.append(dataset_orig)
        except:
            print("no {0} exists".format(experiment))


dataset_orig = collections.OrderedDict()

if len(dataset_list) == 1:
    dataset_orig = dataset_list[0]
else:
    dataset_orig['y'] = np.append(dataset_list[0]['y'], dataset_list[1]['y'])
    dataset_orig['session'] = np.append(dataset_list[0]['session'], dataset_list[1]['session'])
    dataset_orig['list'] = np.append(dataset_list[0]['list'], dataset_list[1]['list'])
    dataset_orig['X'] = orig_dataset

# print(len(dataset_orig['y']))
# dataset_orig['X'] = normalize_sessions(dataset_orig['X'], dataset_orig['session'])
# print dataset_orig['X'].min()
# print dataset_orig['X'].max()
# result = run_loso_xval(dataset_orig, classifier_name = 'L2', search_method = 'rand', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)
# print(result)

subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset_delib.pkl'
dataset = joblib.load(subject_dir)
sessions = np.unique(dataset['session'])
print(sessions)

dataset_enc = select_phase(dataset)
dataset['X'] = normalize_sessions(dataset['X'], dataset['session'])
print(len(dataset_enc['y']))
#dataset_aug = generate_data(dataset, n_repeat=2, sigma_noise=sigma_noise)
dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])
print dataset_enc['X'].min()
print dataset_enc['X'].max()

result = run_loso_xval(dataset_enc, classifier_name = 'L2', search_method = 'rand', type_of_data = 'short',  feature_select= 0,  adjusted = 1, C_factor = 1.0e-4)
print(result)



result = run_loso_xval(dataset_enc, classifier_name = 'current', search_method = 'rand', type_of_data = 'short',  feature_select= 0,  adjusted = 1, C_factor = 1.0e-4)
print(result)