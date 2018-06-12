import numpy as np
from scipy.stats.mstats import zscore
import time
import warnings
from math import sqrt
from random import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib
import collections
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn import linear_model
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from math import sqrt
import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression  # L2
from sklearn.ensemble import RandomForestClassifier as RF  # random forests
from sklearn import svm # svm
#import xgboost as xgb  # xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut # leave one group out
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RandomizedLogisticRegression as RL
from sklearn. preprocessing import minmax_scale


from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import hyperopt
#import xgboost as xgb
import collections
from functools import wraps

#from noisy_classifier_class import*


# timer function
def timethis(func):
    '''
    Decorator that reports execuation time
    :param func: function
    :return: wrapper function
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper



def select_phase(dataset, phase = 'WORD'):

    X = dataset['X']
    y = dataset['y']
    listpos = dataset['list']
    session = dataset['session']
    event_type = dataset['type']

    dataset_select = collections.OrderedDict()
    if (phase == 'ALL'):
        return dataset
    else:
        indices = np.where(event_type == phase)[0]
        dataset_select['X'] = X[indices,:]
        dataset_select['y'] = y[indices]
        dataset_select['list'] = listpos[indices]
        dataset_select['session'] = session[indices]
        dataset_select['type'] = event_type[indices]
        return dataset_select




def map_bipolar_to_region(x):

    brain_region_mapping = {
    'unknown': 'unknown',
    'Left CA1' : 'Hipp',
    'Left Amy' : '',
    'Left CA3' : 'Hipp',
    'Right DG' : 'Hipp',
    'Left DLPFC' : 'MFG',
    'Right CA1' : 'Hipp',
    'Right PCg' : '',
    'Right ACg' : '',
    'Left PRC' : 'MTL',
    'Right EC' : 'MTL',
    'Left PHC' : 'MTL',
    'Right PRC' : 'MTL',
    'Right DLPFC' : 'MFG',
    'Right PHC' : 'MTL',
    'Left TC' : 'TC',
    'Left ACg' : '',
    'Left Sub' : 'Hipp',
    'Left Middle Temporal Gyrus' : 'TC',
    'Right MTL WM' : 'MTL',
    'Right Insula' : '',
    'Left DG' : 'Hipp',
    'supramarginal' : 'IPC',
    'rostralmiddlefrontal': 'MFG',
    'caudalanteriorcingulate': '',
    'temporalpole' : '',
    'parahippocampal' : 'MTL',
    'fusiform' : '',
    'entorhinal' : 'MTL',
    'middletemporal' : 'TC',
    'superiorfrontal' : 'SFG',
    'insula' : '',
    'lingual' : 'OC',
    'caudalmiddlefrontal': 'MFG',
    'bankssts' : 'Speech',
    'inferiortemporal': 'TC',
    'inferiorparietal' : 'IPC',
    'lateraloccipital' : 'OC',
    'superiortemporal' : 'TC',
    'rostralanteriorcingulate' : '',
    'parsopercularis': 'IFG',
    'parsorbitalis': 'IFG',
    'parstriangularis': 'IFG',
    'cuneus': 'OC',
    'pericalcarine':'OC'
    }




    x = str.lower(str(x)).strip()
    brain_region_names = brain_region_mapping.keys()
    brain_region_names = [str.lower(z).strip() for z in brain_region_names]

    #brain_region_mapping = dict((key, value) for key in brain_region_names for value in brain_region_mapping.values())


    splits = str.split(str(x), " ")
    if(len(splits) >1):
        direction = splits[0]
        region_x = splits[1]
    else:
        direction = ""
        region_x = splits[0]


    if x in brain_region_names:
        index = brain_region_names.index(x)
        return direction,brain_region_mapping[brain_region_mapping.keys()[index]]
    elif region_x in brain_region_names:
        index = brain_region_names.index(region_x)

        return direction,brain_region_mapping[brain_region_mapping.keys()[index]]

    else:
        return direction,""   # return empty if no match


def make_frequency(index):
    n_freqs = 178
    return np.arange(index*n_freqs, index*n_freqs + n_freqs)


def normalize_sessions(pow_mat, event_sessions):
    sessions = np.unique(event_sessions)
    for sess in sessions:
        sess_event_mask = (event_sessions == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat

def normalize_sessions(pow_mat, event_sessions):
    sessions = np.unique(event_sessions)
    for sess in sessions:
        sess_event_mask = (event_sessions == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat

def normalize_sessions_by_tasks(pow_mat, event_sessions, event_types):

    encoding_events_mask = (event_types == "WORD")
    retrieval_events_mask =  (event_types != "WORD")

    pow_mat[encoding_events_mask] = normalize_sessions(pow_mat[encoding_events_mask], event_sessions[encoding_events_mask])
    pow_mat[retrieval_events_mask] = normalize_sessions(pow_mat[retrieval_events_mask], event_sessions[retrieval_events_mask])
    return pow_mat


def scale_sessions(pow_mat, event_sessions, pow_mat_ref = None, event_sessions_ref = None):

    sessions = np.unique(event_sessions)
    for sess in sessions:
        sess_event_mask = (event_sessions == sess)
        sess_event_mask_ref = (event_sessions_ref == sess)
        mins = pow_mat_ref[sess_event_mask_ref].min(axis = 0)
        maxs = pow_mat_ref[sess_event_mask_ref].max(axis = 0)
        pow_mat[sess_event_mask] = (pow_mat[sess_event_mask] - mins)

    return pow_mat


# tuning parameters
def opt_params(X,y, session, list_session, classifier_name, ind_params, search_method, type_of_data, feature_select):


    # compute score for each param set
    def hyperopt_train_test(params):
        logo = LeaveOneGroupOut()  # create fold indices
        logo_generator = logo.split(X,y, session)
        #lolo_generator = logo.split(X,y, list_session)
        skf = StratifiedKFold(n_splits = 3)
        cv_generator = skf.get_n_splits(X,y)

        params_joint = ind_params.copy()
        params_joint.update(params)


        if classifier_name == 'RF':
            clf = RF(**params_joint)

        if classifier_name == 'XGB':
            clf = xgb.XGBClassifier(**params_joint)

        if classifier_name == 'SVM':
            clf = svm.SVC(**params_joint)

        if classifier_name == 'L2':
            clf = LogisticRegression(**params_joint)

        if classifier_name == 'noisy_L2':
            clf = noisy_LogisticRegression(**params_joint)

        if classifier_name == 'L1':
            clf = LogisticRegression(**params_joint)


        n_sessions = len(np.unique(session))
        if n_sessions >=2:
            cross_scores = cross_val_score(clf, X,y, cv = logo_generator, scoring = 'roc_auc',  n_jobs = 1)
        else:
            #cross_scores = cross_val_score(clf, X,y, cv = lolo_generator, n_jobs = 10, scoring = 'roc_auc')
            cross_scores = cross_val_score(clf, X,y, cv = cv_generator, n_jobs = 1, scoring = 'roc_auc')
        return cross_scores.mean()

    # define search parameter space
    if classifier_name == 'RF':
        space4classifier ={
              'n_estimators': hp.choice('n_estimators', [500]),
              #'n_estimators': hp.qloguniform('n_estimators', 100,1000,1),
              'max_features': hp.choice('max_features', ['sqrt','log2', 0.2]),
              #'max_features': hp.choice('max_features', np.arange(100, X.shape[1],200)),

              'max_depth': hp.choice('max_depth', np.arange(4,10,step =2)),

              'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(5,20,step = 5))
    }
    if classifier_name == 'XGB':
        space4classifier ={
          'n_estimators': hp.choice('n_estimators', np.arange(50,500, step = 50)),
         #'n_estimators': hp.qloguniform('n_estimators', 5,10,1.0),
          'max_depth': hp.choice('max_depth', np.arange(4,10, 1)),
          'learning_rate': hp.choice('learning_rate',[0.01,0.1]),

          #'reg_lambda': hp.choice('reg_lambda', [0,0.01,0.05,0.1,1.0,10]),
          #'min_child_weight':hp.choice('min_child_weight',np.arange(3.0,7.0, step =2.0)),
          #'scale_pos_weight':hp.uniform('scale_pos_weight',1,10),
          'colsample_bytree': hp.uniform('colsample_bytree',0.5,1.0),
          'subsample': hp.uniform('subsample',0.7,1.0)
    }

    if classifier_name == 'SVM':
        space4classifier ={
          #'C': hp.choice('C', 10.**np.arange(-2,10, step = 1.0)),
          'C': hp.loguniform('C', 0,10),
          'gamma': hp.loguniform('gamma', -5,5)
          #'gamma': hp.choice('gamma', ['auto']),
          #'kernel':hp.choice('kernel', ['rbf', 'poly', 'sigmoid']),
          #'degree':hp.choice('degree', np.arange(3,6))
    }

    if classifier_name == 'noisy_L2':
        space4classifier ={
          #'C': hp.choice('C', 10.**np.arange(-2,10, step = 1.0)),
          'C': hp.loguniform('C', -7,-1),
          'sigma_noise': hp.choice('sigma_noise', [0.01,0.05,0.1]),
          'noise_penalty':hp.loguniform('noise_penalty',-2,2),
          'w_enc':hp.uniform('w_enc',1.0,5.0)
          #'gamma': hp.choice('gamma', ['auto']),
          #'kernel':hp.choice('kernel', ['rbf', 'poly', 'sigmoid']),
          #'degree':hp.choice('degree', np.arange(3,6))
    }

    if classifier_name == 'L2':
         #space4classifier ={'C': hp.choice('C',np.append(10.**np.arange(-5,-1,step =0.25), np.array(7.2e-4)))
         if type_of_data == 'long':
             space4classifier ={
             'C': hp.loguniform('C',-10,-3)}
         elif feature_select == 1:
             space4classifier ={
             'C': hp.loguniform('C',-10,-3)}
         else:
             space4classifier ={
             'C': hp.loguniform('C',-10,0),

             }




    if classifier_name == 'L1':
         #space4classifier ={'C': hp.choice('C',np.append(10.**np.arange(-5,-1,step =0.25), np.array(7.2e-4)))
         if type_of_data == 'long':
             space4classifier ={
             'C': hp.loguniform('C',-15,-7)}
         elif feature_select == 1:
             space4classifier ={
             'C': hp.loguniform('C',-3,-2)}
         else:
             space4classifier ={
             'C': hp.loguniform('C',-3,-2)}



    global best
    best = 0.3
    def f(params):
            global best
            acc = hyperopt_train_test(params)
            if acc > best:
                best = acc
                print 'new best:', best, params
            return {'loss': -acc, 'status': STATUS_OK}

    #trials = Trials()  # saving trials
    if search_method == 'rand':
        best_params = fmin(f, space4classifier, algo = hyperopt.rand.suggest, max_evals = 2)

    if search_method == 'tpe':
        best_params = fmin(f, space4classifier, algo = hyperopt.tpe.suggest, max_evals = 2)

    return space_eval(space4classifier,best_params)

@timethis
def run_loso_xval(dataset, classifier_name = 'current', search_method = 'rand', type_of_data = 'normal',  feature_select= 0,  adjusted = 1, C_factor = 1.0, **kwargs):

    print C_factor

    # Augmented dataset
    recalls = dataset['y']
    event_sessions = dataset['session']
    list_sessions = dataset['list']
    pow_mat = dataset['X']



    probs = np.empty_like(recalls, dtype=np.float)
    sessions = np.unique(event_sessions)
    probs_sessions = pd.DataFrame()
    auc_session = np.zeros(len(sessions))
    total_elec = int(pow_mat.shape[1]/8)

    N_frequency = 8 # currently fixed at 8 but feel free to adjust
    scores = np.zeros(shape = (N_frequency, total_elec))

    for i,sess in enumerate(sessions):
        print "session " + str(sess) + "..."


        # training set
        probs_temp = np.empty_like(recalls, dtype = np.float)
        insample_mask = (event_sessions != sess)
        insample_pow_mat = pow_mat[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_sess = event_sessions[insample_mask]
        insample_list = list_sessions[insample_mask]

        # test data
        outsample_mask = ~insample_mask
        outsample_pow_mat = pow_mat[outsample_mask]
        outsample_recalls = recalls[outsample_mask]

        indices = np.where(insample_recalls ==1)[0]
        # scaling for class imbalance
        n_recalls = len(indices)
        n_non_recalls = len(insample_recalls) - n_recalls
        # pos_weight = len(insample_recalls)/2.0/n_recalls
        # neg_weight = len(insample_recalls)/2.0/n_non_recalls

        if classifier_name == 'L2':
            ind_params = {'class_weight':'balanced', 'solver':'liblinear'}
            print ind_params
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            ind_params.update(best_params)
            classifier = LogisticRegression(**ind_params)

        if classifier_name == 'L1':
            ind_params = {'class_weight':'balanced', 'solver':'liblinear'}
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            ind_params.update(best_params)
            classifier = LogisticRegression(**ind_params)



        # SVM
        if classifier_name == 'SVM':

            ind_params =  {'class_weight': {0:neg_weight,1:pos_weight}}
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            ind_params.update(best_params)
            ind_params.update({'probability': True})
            classifier = svm.SVC(**ind_params)


        # gradient boosting
        if classifier_name == 'XGB':
            n_recalls = len(np.where(insample_recalls ==1)[0])
            n_non_recalls = len(insample_recalls) - n_recalls
            ind_params = {'objective':'binary:logistic', 'nthread':20}
            ind_params.update({'scale_pos_weight':n_non_recalls/n_recalls})
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            #ind_params.update({'scale_pos_weight':n_non_recalls/n_recalls})
            ind_params.update(best_params)
            classifier = xgb.XGBClassifier(**ind_params)

        # Random Forest
        if classifier_name == 'RF':
            #ind_params = {'class_weight': {0:neg_weight,1:pos_weight}, 'n_jobs':20}
            ind_params = {'class_weight': 'balanced', 'n_jobs':20}
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select)
            ind_params.update(best_params)
            classifier = RF(**ind_params)

        if classifier_name == 'noisy_L2':

            if len(kwargs) > 0:
                sigma_noise = kwargs['sigma_noise']
                noise_penalty = kwargs['alpha']
                C = kwargs['C']
                batch_size = kwargs['batch_size']
                ind_params = {'learning_rate':0.05, 'max_iter':2000, 'batch_size':batch_size, 'momentum':0.8}
                best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select)
                ind_params.update(best_params)

            else:
                ind_params = {'learning_rate':0.05, 'max_iter':2000, 'batch_size':batch_size, 'momentum':0.8, 'C':7.2e-3,'noise_penalty':1.0, 'sigma_noise':0.05}
            #best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select)
            #ind_params.update(best_params)
            classifier = noisy_LogisticRegression(**ind_params)

        if classifier_name == 'current':
            C = 7.2e-4 # adjust for number of electrodes
            if (adjusted):
                C = 7.2e-4*C_factor # adjust for number of electrodes

            penalty_type = 'l2'
            classifier = LogisticRegression(C=C, penalty=penalty_type, class_weight='balanced', solver='liblinear')






        classifier.fit(insample_pow_mat, insample_recalls)
        #print classifier.classes_


        outsample_probs = classifier.predict_proba(outsample_pow_mat)[:,1]
        insample_probs = classifier.predict_proba(insample_pow_mat)[:,1]

        probs[outsample_mask] = outsample_probs
        probs_temp[outsample_mask] = outsample_probs
        probs_temp[insample_mask] = insample_probs
        probs_sessions[str(sess)] = probs_temp


        # save results
        result_sess = collections.OrderedDict()
        result_sess['y'] = insample_recalls
        result_sess['prob'] = insample_probs

        outsample_recalls = recalls[outsample_mask]

        # if classifier_name == 'SVM':
        #     outsample_recalls[outsample_recalls == -1] = 0  # change 0 to -1 for SVM

        #print(outsample_recalls)

        auc_session[i]=sklearn.metrics.roc_auc_score(outsample_recalls,outsample_probs)
        #print 'auc session: ' + str(sess), auc_session[i]

    # combined auc
    auc = sklearn.metrics.roc_auc_score(recalls,probs)
    return {'comb':auc, 'avg':np.mean(auc_session)}





def get_fr_sample_weights(recall, event_type, encoding_multiplier):
    """ Create sample weights based on FR scheme.

    Parameters
    ----------
    events: np.recarrary
        All encoding and retrieval events for consideration in weighting
    encoding_multiplier: float
        Factor determining how much more encoding samples should be weighted
        than retrieval

    Returns
    -------
    weights: np.ndarray
        Sample-level weights

    Notes
    -----
    This function asssumes that the given events are in 'normalized' form,
    i.e. they have already been cleaned and homogenized. By the time events
    are passed to this function, intrusions should have already been removed
    and there should only be 'REC_EVENT' event types for the retrieval
    period. Baseline retrievals are non-recalled REC_EVENTs and actual
    retrievals are recalled REC_EVENTs

    """
    enc_mask = (event_type == 'WORD')
    retrieval_mask = (event_type == 'REC_EVENT')

    n_enc_0 = np.sum(enc_mask & (recall == 0))
    n_enc_1 = np.sum(enc_mask & (recall == 1))
    n_ret_0 = np.sum(retrieval_mask & (recall == 0))
    n_ret_1 = np.sum(retrieval_mask & (recall == 1))


    n_vec = np.array([1.0/n_enc_0, 1.0/n_enc_1, 1.0/n_ret_0, 1.0/n_ret_1 ],
                     dtype=np.float)

    n_vec /= np.mean(n_vec)

    n_vec[:2] *= encoding_multiplier

    n_vec /= np.mean(n_vec)

    # Initialize observation weights to 1
    weights = np.ones(len(recall), dtype=np.float)

    weights[enc_mask & (recall == 0)] = n_vec[0]
    weights[enc_mask & (recall == 1)] = n_vec[1]
    weights[retrieval_mask & (recall == 0)] = n_vec[2]
    weights[retrieval_mask & (recall == 1)] = n_vec[3]

    return weights



def get_prior_sample_weights(y_prior, y_current):

    n_prior = len(y_prior)
    n_current = len(y_current)

    weights = np.concatenate([np.repeat(1.0/n_prior, n_prior), np.repeat(1.0/n_current, n_current)])

    return weights

def get_sample_weights_fr(y):

    n = len(y)
    weights = np.zeros(n)
    recall_mask = y == 1;
    n_recall = np.sum(recall_mask)
    n_non_recall = n - n_recall

    weights[recall_mask] = 1.0*n/n_recall
    weights[~recall_mask] = 1.0*n/n_non_recall

    return weights


# combined classifier
@timethis
def run_loso_xval_comb(dataset, dataset_enc, classifier_name = 'current', search_method = 'rand', type_of_data = 'normal',  feature_select= 0,  adjusted = 1, C_factor = 1.0, **kwargs):

    print C_factor

    # Augmented dataset
    recalls = dataset['y']
    event_sessions = dataset['session']
    list_sessions = dataset['list']
    pow_mat = dataset['X']

    probs = np.empty_like(dataset_enc['y'], dtype=np.float)  # encoding events only


    sessions = np.unique(event_sessions)
    probs_sessions = pd.DataFrame()
    auc_session = np.zeros(len(sessions))
    total_elec = int(pow_mat.shape[1]/8)

    N_frequency = 8 # currently fixed at 8 but feel free to adjust
    scores = np.zeros(shape = (N_frequency, total_elec))

    for i,sess in enumerate(sessions):
        print "session " + str(sess) + "..."


        # training set
        probs_temp = np.empty_like(recalls, dtype = np.float)
        insample_mask = (event_sessions != sess)
        insample_pow_mat = pow_mat[insample_mask]
        insample_recalls = recalls[insample_mask]
        insample_sess = event_sessions[insample_mask]
        insample_list = list_sessions[insample_mask]
        event_type = dataset['type'][insample_mask]



        indices = np.where(insample_recalls ==1)[0]
        # scaling for class imbalance
        n_recalls = len(indices)
        n_non_recalls = len(insample_recalls) - n_recalls
        # pos_weight = len(insample_recalls)/2.0/n_recalls
        # neg_weight = len(insample_recalls)/2.0/n_non_recalls

        if classifier_name == 'L2':
            ind_params = {'class_weight':'balanced', 'solver':'liblinear'}
            print ind_params
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            ind_params.update(best_params)
            classifier = LogisticRegression(**ind_params)

        if classifier_name == 'L1':
            ind_params = {'class_weight':'balanced', 'solver':'liblinear'}
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            ind_params.update(best_params)
            classifier = LogisticRegression(**ind_params)



        # SVM
        if classifier_name == 'SVM':

            ind_params =  {'class_weight': {0:neg_weight,1:pos_weight}}
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            ind_params.update(best_params)
            ind_params.update({'probability': True})
            classifier = svm.SVC(**ind_params)


        # gradient boosting
        if classifier_name == 'XGB':
            n_recalls = len(np.where(insample_recalls ==1)[0])
            n_non_recalls = len(insample_recalls) - n_recalls
            ind_params = {'objective':'binary:logistic', 'nthread':20}
            ind_params.update({'scale_pos_weight':n_non_recalls/n_recalls})
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select) # tuning parameters
            #ind_params.update({'scale_pos_weight':n_non_recalls/n_recalls})
            ind_params.update(best_params)
            classifier = xgb.XGBClassifier(**ind_params)

        # Random Forest
        if classifier_name == 'RF':
            #ind_params = {'class_weight': {0:neg_weight,1:pos_weight}, 'n_jobs':20}
            ind_params = {'class_weight': 'balanced', 'n_jobs':20}
            best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select)
            ind_params.update(best_params)
            classifier = RF(**ind_params)

        if classifier_name == 'noisy_L2':
            print kwargs
            if len(kwargs) > 0:

                C = kwargs['C']
                batch_size = kwargs['batch_size']
                ind_params = {'class_weight' : 'balanced',  'C':C, 'solver':'liblinear'}
                #best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select)
                #ind_params.update(best_params)

            else:
                ind_params = {'learning_rate':0.01, 'max_iter':2000, 'batch_size':batch_size, 'momentum':0.8, 'C':7.2e-3, 'sigma_noise':0.05}
            #best_params = opt_params(insample_pow_mat,insample_recalls, insample_sess, insample_list, classifier_name, ind_params, search_method, type_of_data, feature_select)
            #ind_params.update(best_params)
            #classifier = noisy_LogisticRegression(**ind_params)
            classifier = LogisticRegression(**ind_params)

        if classifier_name == 'current':
            C = 7.2e-4 # adjust for number of electrodes
            if (adjusted):
                C = 7.2e-4*C_factor # adjust for number of electrodes

            penalty_type = 'l2'
            classifier = LogisticRegression(C=C, penalty=penalty_type, class_weight='balanced', solver='liblinear')


                # test data

        outsample_mask = (dataset_enc['session'] == sess)
        outsample_pow_mat = dataset_enc['X'][outsample_mask]
        outsample_recalls = dataset_enc['y'][outsample_mask]


        # test data
        encoding_mutiplier = kwargs['w_enc']
        weights = get_fr_sample_weights(insample_recalls, event_type, encoding_multiplier= encoding_mutiplier)


        classifier.fit(insample_pow_mat, insample_recalls, sample_weight = weights)
        #print classifier.classes_


        outsample_probs = classifier.predict_proba(outsample_pow_mat)[:,1]
        insample_probs = classifier.predict_proba(insample_pow_mat)[:,1]

        probs[outsample_mask] = outsample_probs
        # probs_temp[outsample_mask] = outsample_probs
        # probs_temp[insample_mask] = insample_probs
        # probs_sessions[str(sess)] = probs_temp

        #
        # #save results
        # result_sess = collections.OrderedDict()
        # result_sess['y'] = insample_recalls
        # result_sess['prob'] = insample_probs

        #outsample_recalls = recalls[outsample_mask]

        if classifier_name == 'SVM':
            outsample_recalls[outsample_recalls == -1] = 0  # change 0 to -1 for SVM



        auc_session[i]=sklearn.metrics.roc_auc_score(outsample_recalls,outsample_probs)
        print 'auc session: ' + str(sess), auc_session[i]

    # combined auc


    recalls_enc = dataset_enc['y']
    auc = sklearn.metrics.roc_auc_score(recalls_enc,probs)
    return {'comb':auc, 'avg':np.mean(auc_session)}
