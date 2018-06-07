

# create time-series X object from raw data
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
from classifier import*
import sys
from helper_funcs import*
args = sys.argv
index = int(args[1])


rhino_root = '/Volumes/RHINO'
dir1 = rhino_root + '/scratch/cdl/supereeg/events'
subjects1 = [str.split(x, '_')[0] for x in os.listdir(dir1)]

dir1 = rhino_root + '/scratch/lucy.owen/supereeg/events'
subjects2 = [str.split(x, '_')[0] for x in os.listdir(dir1)]

subjects = np.unique(subjects1 + subjects2)
subjects = [x for x in subjects if x!='bad']

subject = subjects[index]
print(subject)

pow_mat_orig = []
pow_mat_recon = []


for experiment in ['RAM_FR1', 'RAM_CatFR1']:
    try:

        try:
            subject_dir = rhino_root + '/scratch/cdl/supereeg/events/' + subject + '_' + experiment + '/'
            recon_files = glob.glob1(subject_dir, "*_recon.bo")
            orig_files = glob.glob1(subject_dir, "*[0-9].bo")
            fname_events = subject_dir + 'events.npy'
            events = np.load(fname_events)
            word_events = events[events['type'] == 'WORD']
            roots = np.unique(word_events['eegfile'])
            root = rhino_root + roots[0]
            p_reader = ParamsReader(dataroot = root)
            params = p_reader.read()
            bo_recon_vec = []
            epsilon = 1.0e-3
        except:
            print("not in cdl")

        try:
            subject_dir = rhino_root + '/scratch/lucy.owen/supereeg/events/' + subject + '_' + experiment + '/'
            recon_files = glob.glob1(subject_dir, "*_recon.bo")
            orig_files = glob.glob1(subject_dir, "*[0-9].bo")
            fname_events = subject_dir + 'events.npy'
            events = np.load(fname_events)
            word_events = events[events['type'] == 'WORD']
            roots = np.unique(word_events['eegfile'])
            root = rhino_root + roots[0]
            p_reader = ParamsReader(dataroot = root)
            params = p_reader.read()
            bo_recon_vec = []
            epsilon = 1.0e-3
        except:
            print("not in lucy.owen")



        t = time.time()
        pow_vec_orig = []
        pow_vec_recon = []
        print(len(recon_files))
        print(len(orig_files))
        print (recon_files)
        assert(len(recon_files) == len(events))

        for i, (file_recon, file_orig) in enumerate(zip(recon_files, orig_files)):
            if i%100 == 0:
                print(i)
            fname_recon = subject_dir + file_recon
            fname_orig = subject_dir + file_orig
            bo_orig = se.load(fname_orig)
            bo_recon = se.load(fname_recon)
            data_orig = create_time_seriesX_from_superEEG(bo_orig)
            data_recon = create_time_seriesX_from_superEEG(bo_recon)

            pow_wavelet_orig = compute_log_power(data_orig)
            pow_wavelet_recon = compute_log_power(data_recon)
            pow_vec_orig.append(pow_wavelet_orig)
            pow_vec_recon.append(pow_wavelet_recon)

        t_run = time.time()-t

        pow_vec_orig = np.stack(pow_vec_orig)
        pow_vec_recon = np.stack(pow_vec_recon)

        pow_vec_orig = pow_vec_orig.reshape((pow_vec_orig.shape[0], pow_vec_orig.shape[1]*pow_vec_orig.shape[2]))
        pow_vec_recon = pow_vec_recon.reshape((pow_vec_recon.shape[0], pow_vec_recon.shape[1]*pow_vec_recon.shape[2]))

        pow_mat_recon.append(pow_vec_recon)
        pow_mat_orig.append(pow_vec_orig)

    except:
        print("experiement {0} does not exist".format(experiment))

pow_mat_orig = np.concatenate(pow_mat_orig, axis = 0)
pow_mat_recon = np.concatenate(pow_mat_recon, axis = 0)

import h5py
save_dir = rhino_root + '/scratch/tphan/superautoencoder/' +  subject +   '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = h5py.File(save_dir + '/power.hdf5', 'w')
f.create_dataset('orig', data = pow_mat_orig, dtype = 'f')
f.create_dataset('recon', data = pow_mat_recon, dtype = 'f')
f.close()




#f = h5py.File('mytestfile.hdf5', 'r')
#
# subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset_delib.pkl'
# dataset = joblib.load(subject_dir)
# sessions = np.unique(dataset['session'])
# print(sessions)
#
# dataset_enc = select_phase(dataset)
# dataset['X'] = normalize_sessions(dataset['X'], dataset['session'])
# #dataset_aug = generate_data(dataset, n_repeat=2, sigma_noise=sigma_noise)
# dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])
# result = run_loso_xval(dataset_enc, classifier_name = 'current', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)
#
# dataset_enc['X'] = pow_mat_recon
# dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])
# result_mono = run_loso_xval(dataset_enc, classifier_name = 'current', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)


recon_files
event_numbers =np.array([str.split(x, "_")[0] for x in recon_files], dtype = int)
a = np.arange(1800)
for i, num in enumerate(a):
    if num not in event_numbers:
        print(num)