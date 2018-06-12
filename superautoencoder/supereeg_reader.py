

# create time-series X object from raw data
import glob
import os
import sys
import time

import supereeg as se
import xarray as xr
from classifier import*
from ptsa.data.readers import JsonIndexReader
from ptsa.data.readers import ParamsReader
from ptsa.data.readers import TalReader

from superautoencoder.helper_funcs import*

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



#
# import h5py
# save_dir_file = rhino_root + '/scratch/tphan/superautoencoder/' +  subject +   '/power_test.hdf5'
# if os.path.exists(save_dir_file):
# 	print("removing {0} data".format(subject))
# 	os.remove(save_dir_file)

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

        assert(len(recon_files) == len(events))

        experiment = 'FR1'
        jr = JsonIndexReader(rhino_root + '/protocols/r1.json')  # Build JSON reader
        pairs_path = jr.get_value('pairs', subject=subject, experiment=experiment)
        tal_reader = TalReader(filename=pairs_path)
        monopolar_channels = tal_reader.get_monopolar_channels()
        bipolar_channels = tal_reader.get_bipolar_pairs()

        recon_files = np.sort(recon_files)
        orig_files = np.sort(orig_files)

        data_orig_vec = []
        data_recon_vec =[]


        for i, (recon_file, orig_file) in enumerate(zip(recon_files, orig_files)):
            if i%100 == 0:
                print(i)
            fname_recon = subject_dir + recon_file
            fname_orig = subject_dir + orig_file
            bo_orig = se.load(fname_orig)
            bo_recon = se.load(fname_recon)
            data_orig = create_time_seriesX_from_superEEG(bo_orig)

            data_orig['channels'] = monopolar_channels
            #data_orig = MonopolarToBipolarMapper(time_series = data_orig, bipolar_pairs = bipolar_channels).filter()
            data_recon = create_time_seriesX_from_superEEG(bo_recon)

            ### Testing part

            data_orig_coords = list(zip(data_orig.attrs['x'], data_orig.attrs['y'], data_orig.attrs['z']))
            data_recon_coords = list(zip(data_recon.attrs['x'], data_recon.attrs['y'], data_recon.attrs['z']))

            data_orig_vec.append(data_orig)
            data_recon_vec.append(data_recon_vec)


        data_orig_array = concat_time_seriesX(data_orig_vec)


            # indices = [i for (i,x) in enumerate(data_recon_coords) if x in data_orig_coords]
            # data_recon[indices[0]].values
            # data_orig[data_orig_coords == data_recon_coords[indices[0]]].values
            #pow_wavelet_orig = compute_log_power(data_orig)
            #pow_wavelet_recon = compute_log_power(data_recon)
            #pow_vec_orig.append(pow_wavelet_orig)
            #pow_vec_recon.append(pow_wavelet_recon)

        # t_run = time.time()-t
        #
        # pow_vec_orig = np.stack(pow_vec_orig)
        # pow_vec_recon = np.stack(pow_vec_recon)
        #
        # pow_vec_orig = pow_vec_orig.reshape((pow_vec_orig.shape[0], pow_vec_orig.shape[1]*pow_vec_orig.shape[2]))
        # pow_vec_recon = pow_vec_recon.reshape((pow_vec_recon.shape[0], pow_vec_recon.shape[1]*pow_vec_recon.shape[2]))
        #
        # pow_mat_recon.append(pow_vec_recon)
        # pow_mat_orig.append(pow_vec_orig)

    except:
        print("experiement {0} does not exist".format(experiment))

pow_mat_orig = np.concatenate(pow_mat_orig, axis = 0)
pow_mat_recon = np.concatenate(pow_mat_recon, axis = 0)

import h5py
save_dir = rhino_root + '/scratch/tphan/superautoencoder/' +  subject +   '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = h5py.File(save_dir + '/power_test.hdf5', 'w')
f.create_dataset('orig', data = pow_mat_orig, dtype = 'f')
#f.create_dataset('recon', data = pow_mat_recon, dtype = 'f')
f.close()


arr = xr.DataArray(np.random.randn(2, 3),[('x', ['a', 'b']), ('y', [10, 20, 30])])