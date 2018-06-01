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


coords.update(mni_coords)
coords = OrderedDict()
coords.update(times)
coords.update(channels)
coords.update(samplerate)



wf = MorletWaveletFilterCpp(time_series=data.T, freqs=np.logspace(np.log10(3), np.log10(180), 8), output='power', cpus = 25)
pow_wavelet, phase_wavelet = wf.filter()
pow_wavelet = np.log10(pow_wavelet)


pow_wavelet = pow_wavelet.remove_buffer(duration=buffer)



# grab all _recon.bo file
rhino_root = '/Volumes/RHINO'
subject_dir = rhino_root + '/scratch/cdl/supereeg/events/' + subject + '_RAM_FR1/'

recon_files = glob.glob1(subject_dir, "*_recon.bo")
original_files = glob.glob1(subject_dir, "*[0-9].bo")


rhino_root = '/Volumes/RHINO'
subject = 'R1135E'
subject_dir = rhino_root + '/scratch/cdl/supereeg/events/' + subject + '_RAM_FR1/'
fname_events = subject_dir + 'events.npy'
events = np.load(fname_events)
word_events = events[events['type'] == 'WORD']
roots = np.unique(word_events['eegfile'])
root = rhino_root + roots[0]
p_reader = ParamsReader(dataroot = root)
params = p_reader.read()
bo_recon_vec = []
epsilon = 10e-4

t = time.time()

pow_vec = []
for i, file in enumerate(recon_files[:10]):
    print(i)
    fname_recon = subject_dir + file
    bo_recon = se.load(fname_recon)
    times = {'time':np.arange(bo_recon.data.shape[0])*1000.0/params['samplerate']}
    samplerate = {'samplerate':params['samplerate']}
    mni_coords = bo_recon.locs.to_dict('series')
    channels = {'channel':np.arange(bo_recon.data.shape[1])}
    coords = {**times, **channels, **samplerate}
    data = TimeSeriesX(bo_recon.data, coords = coords, dims = ['time', 'channel'])
    data.attrs['mni'] = mni_coords
    wf = MorletWaveletFilterCpp(time_series=data.T, freqs=np.logspace(np.log10(3), np.log10(180), 8), output='power', cpus = 25)
    pow_wavelet_recon, phase_wavelet = wf.filter()
    pow_wavelet_recon = np.log10(pow_wavelet_recon + epsilon)
    pow_wavelet_avg = pow_wavelet_recon.mean(axis = 2)
    pow_vec.append(pow_wavelet_avg)


t_run = time.time()-t
