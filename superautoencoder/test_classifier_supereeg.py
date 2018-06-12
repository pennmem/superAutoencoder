#import supereeg as se
import os

from sklearn.externals import joblib
#from classifier_randomized import*
from superautoencoder.classifier import*

import sys
#from helper_funcs import*
import h5py

args = sys.argv
index = int(args[1])

rhino_root = '/Volumes/RHINO'
dir = rhino_root + '/scratch/tphan/superautoencoder/'
subjects = np.sort(os.listdir(dir))

subject = subjects[index]

subject = 'R1065J'

# save_dir = rhino_root + '/scratch/tphan/superautoencoder/' + subject + '/'
# save_file = save_dir + 'result_supereeg.pkl'
# if os.path.exists(save_file):
# 	print("removing result ...")
# 	os.remove(save_file)

print(subject)
subject_dir = dir + subject + '/power_test.hdf5'
f = h5py.File(subject_dir, 'r')
orig_dataset = f['orig'].value
supereeg_dataset = f['recon'].value
#f.close()

subject_dir = dir + subject + '/features_test.hdf5'
f2 = h5py.File(subject_dir, 'r')
y = f2['y']

print y.value
session = f2['session']
list = f2['list']
#f2.close()

#print("number of features ", supereeg_dataset.shape)
dataset_list = []
dataset_orig = collections.OrderedDict()
dataset_orig['X'] = orig_dataset
dataset_orig['y'] = y.value
dataset_orig['session'] = session.value
dataset_orig['list'] = list.value

subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset_delib.pkl'
dataset = joblib.load(subject_dir)
sessions = np.unique(dataset['session'])
print("number of sessions", np.unique(dataset_orig['session']))

dataset_orig['X'] = normalize_sessions(dataset_orig['X'], dataset_orig['session'])
dataset_enc = select_phase(dataset)
dataset['X'] = normalize_sessions(dataset['X'], dataset['session'])
result_orig = run_loso_xval(dataset_orig, classifier_name = 'L2', search_method = 'rand', type_of_data = 'short',  feature_select= 0,  adjusted = 1, C_factor = 1.0)

dataset_supereeg = dataset_orig
dataset_supereeg['X'] = supereeg_dataset
dataset_supereeg['X'] = normalize_sessions(dataset_supereeg['X'], dataset_supereeg['session'])
result_supereeg = run_loso_xval(dataset_supereeg, classifier_name = 'L2', search_method = 'rand', type_of_data = 'long',  feature_select= 0,  adjusted = 1, C_factor = 1.0)
result_current = run_loso_xval(dataset_enc, classifier_name = 'current', search_method = 'rand', type_of_data = 'short',  feature_select= 0,  adjusted = 1, C_factor = 1.0)


save_dir = rhino_root + '/scratch/tphan/superautoencoder/' + subject + '/'

result_all = collections.OrderedDict()
result_all['mono'] = result_orig
result_all['supereeg'] = result_supereeg
result_all['current'] = result_current
result_all['complete'] = dataset_orig['X'].shape[0] == dataset_enc['X'].shape[0]
#result_all['L2'] = result_L2
joblib.dump(result_all, save_dir + '/result_random_search_supereeg.pkl')
print result_all

f.close()
f2.close()


