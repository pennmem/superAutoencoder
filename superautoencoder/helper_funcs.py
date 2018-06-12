# create time-series X object from raw data
import numpy as np
from ptsa.data.TimeSeriesX import TimeSeriesX
from ptsa.data.filters import (
    MorletWaveletFilterCpp)

def create_time_seriesX_from_superEEG(data, samplerate = 500.0):
    times = {'time':np.arange(data.data.shape[0])*1000.0/samplerate}
    samplerate = {'samplerate':samplerate}
    mni_coords = data.locs.to_dict('series')
    channels = {'channels':np.arange(data.data.shape[1])}
    coords = {**times, **channels, **samplerate}
    data = TimeSeriesX(data.data, coords = coords, dims = ['time', 'channels'])
    data.attrs['x'] = mni_coords['x']
    data.attrs['y'] = mni_coords['y']
    data.attrs['z'] = mni_coords['z']
    return data

def concat_time_seriesX(data_vec):
    n_events = len(data_vec)
    times = {'time':data_vec[0]['time'].values}
    channels = {'channels':data_vec[0]['channels'].values}
    samplerate = {'samplerate':data_vec[0]['samplerate']}
    events = {'event':np.arange(n_events)}
    coords = {**times, **channels, **samplerate, **events}

    data_vec_values = [x.values for x in data_vec]
    data_array = np.stack(data_vec_values)
    data = TimeSeriesX(data_array, coords = coords, dims = ['event','time', 'channels'])

    data.attrs['x'] = data_vec[0].attrs['x']
    data.attrs['y'] = data_vec[0].attrs['y']
    data.attrs['z'] = data_vec[0].attrs['z']

    return data



def compute_log_power(data, frequencies = np.logspace(np.log10(3), np.log10(180), 8), epsilon = 1.0e-4, buffer = 2.0):
    wf = MorletWaveletFilterCpp(time_series=data.T, freqs=frequencies, output='power', cpus = 25)
    pow_wavelet, phase_wavelet = wf.filter()
    pow_wavelet = pow_wavelet.remove_buffer(duration = buffer)
    pow_wavelet = np.log10(pow_wavelet + epsilon)

    pow_wavelet_avg = pow_wavelet.mean(axis = 2)
    return pow_wavelet_avg
