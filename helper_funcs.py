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



def create_time_seriesX_from_superEEG(data, samplerate = 500.0):
    times = {'time':np.arange(data.data.shape[0])*1000.0/samplerate}
    samplerate = {'samplerate':samplerate}
    mni_coords = data.locs.to_dict('series')
    channels = {'channel':np.arange(data.data.shape[1])}
    coords = {**times, **channels, **samplerate}
    data = TimeSeriesX(data.data, coords = coords, dims = ['time', 'channel'])
    data.attrs['x'] = mni_coords['x']
    data.attrs['y'] = mni_coords['y']
    data.attrs['z'] = mni_coords['z']
    return data

def compute_log_power(data, frequencies = np.logspace(np.log10(3), np.log10(180), 8), epislon = 1.0e-4, buffer = 2.0):
    wf = MorletWaveletFilterCpp(time_series=data.T, freqs=frequencies, output='power', cpus = 25)
    pow_wavelet, phase_wavelet = wf.filter()
    pow_wavelet = pow_wavelet.remove_buffer(duration = buffer)
    pow_wavelet = np.log10(pow_wavelet + epsilon)

    pow_wavelet_avg = pow_wavelet.mean(axis = 2)
    return pow_wavelet_avg




rhino_root = '/Volumes/RHINO'
subject = 'R1135E'

pow_mat_exp = []
pow_mat_recon = []
for experiment in ['RAM_FR1', 'RAM_CatFR1']:
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

    t = time.time()
    pow_vec_orig = []
    pow_vec_recon = []

    for i, (file_recon, file_orig) in enumerate(zip(recon_files, orig_files)):
        if i%100 == 0:
            print(i)
        fname_recon = subject_dir + file_recon
        fname_orig = subject_dir + file_orig
        bo_orig = se.load(fname_orig)
        bo_recon = se.load(fname_recon)
        #data_orig = create_time_seriesX_from_superEEG(bo_orig)
        data_recon = create_time_seriesX_from_superEEG(bo_recon)

        #pow_wavelet_orig = compute_log_power(data_orig)
        pow_wavelet_recon = compute_log_power(data_recon)
        #pow_vec_orig.append(pow_wavelet_orig)
        pow_vec_recon.append(pow_wavelet_recon)

    t_run = time.time()-t

    pow_mat = np.stack(pow_vec_recon)
    pow_mat = pow_mat.reshape((pow_mat.shape[0], pow_mat.shape[1]*pow_mat.shape[2]))
    pow_mat_recon.append(pow_mat)

pow_mat = np.concatenate(pow_mat_exp, axis = 0)

subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset_delib.pkl'
dataset = joblib.load(subject_dir)
sessions = np.unique(dataset['session'])
print(sessions)

dataset_enc = select_phase(dataset)
dataset['X'] = normalize_sessions(dataset['X'], dataset['session'])
#dataset_aug = generate_data(dataset, n_repeat=2, sigma_noise=sigma_noise)
dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])
result = run_loso_xval(dataset_enc, classifier_name = 'current', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)

dataset_enc['X'] = pow_mat
dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])
result_mono = run_loso_xval(dataset_enc, classifier_name = 'L2', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)
