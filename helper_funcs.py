
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
