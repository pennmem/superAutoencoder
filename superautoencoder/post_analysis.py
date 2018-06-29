# build a joint classifier using encoding and retrieval data
# version uses 222 subjects

import os
import numpy as np
from sklearn.externals import joblib


rhino_root = '/Volumes/RHINO'
all_subjects = np.sort(os.listdir(rhino_root + '/scratch/tphan/superautoencoder/'))
#from superautoencoder import*


auc_orig = []
auc_auto = []

subject_list = []



for subject in all_subjects:
    try:
        subject_dir_save = rhino_root + '/scratch/tphan/superautoencoder/' + subject + '/aae_result_sigmoid.pkl'
        result = joblib.load(subject_dir_save)
        auc_orig.append(result['current']['comb'])
        auc_auto.append(result['aae'])
        subject_list.append(subject)

    except:
        print "subject = ", subject


# testing
auc_current = np.array(auc_orig)
auc_auto = np.array(auc_auto)
subject_list = np.array(subject_list)
from scipy.stats import ttest_1samp
delta_auc = auc_auto - auc_current
ttest_1samp(delta_auc,0)


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
result_frame.to_csv('auc_table.csv')