# build a joint classifier using encoding and retrieval data
# version uses 222 subjects

import os

from sklearn.externals import joblib

from superautoencoder.classifier import*

rhino_root = '/Volumes/RHINO'
all_subjects = np.sort(os.listdir(rhino_root + '/scratch/tphan/superautoencoder/'))

auc_orig = []
auc_supereeg = []
auc_current = []
auc_L2 = []

subject_list = []



for subject in all_subjects:
    try:
        subject_dir_save = rhino_root + '/scratch/tphan/superautoencoder/' + subject + '/dataset_supereeg_result.pkl'
        result = joblib.load(subject_dir_save)
        auc_orig.append(result['mono']['comb'])
        auc_supereeg.append(result['supereeg']['comb'])
        auc_current.append(result['current']['comb'])
        auc_L2.append(result['L2']['comb'])
        subject_list.append(subject)

    except:
        print "subject = ", subject


# testing
auc_current = np.array(auc_current)
auc_supereeg = np.array(auc_supereeg)
auc_L2 = np.array(auc_L2)
auc_orig = np.array(auc_orig)
subject_list = np.array(subject_list)
from scipy.stats import ttest_1samp


auc_noise = np.array(auc_noise)
auc_current = np.array(auc_current)
delta_auc = auc_noise- auc_current
subject_list = np.array(subject_list)


test =ttest_1samp(delta_auc,0)

import seaborn as sns
from matplotlib import pyplot as plt
fig,ax = plt.subplots(1,1, figsize = (10,6))
ax = sns.distplot(delta_auc, rug = True, kde_kws={"lw":3})
ax.set_xlabel('$\Delta$ AUC', size = 15)
ax.set_title('Reopt - Current', size = 15)
ax.tick_params(axis = 'both', labelsize = 15)
ax.axvline(np.mean(delta_auc), color = 'red', lw = 3)
ax.text( -0.02,  30,  str('p = 0.04') , size = 12)
ax.text( -0.02,  40,  '$\Delta = $' + str(round(np.mean(delta_auc),4)) , size = 12)

fig.savefig('joint_reopt_result.pdf', dpi = 1000)

result_frame = pd.DataFrame({'joint_current' : auc_current, 'joint_reopt' : auc_noise, 'subject':subject_list})
result_frame.to_csv('auc_table.csv')--