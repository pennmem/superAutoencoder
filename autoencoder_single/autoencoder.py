import numpy as np
import ptsa
import ramutils
import os
import pandas as pd
import feather
import os
import glob
import pandas as pd
import sys
from sklearn.externals import joblib
#from classifier_noisy import*
#from classifier import*
import collections
#from autoencoder import*
from keras import models
from keras import layers
from keras.layers import LeakyReLU

from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import sys
sys.path.append(os.getcwd() + '/autoencoder_single')
import helper_funcs
from helper_funcs import*
import classifier
from classifier import*


args = sys.argv
index = int(args[1])
denoising = args[2]

rhino_root = '/Volumes/RHINO'
all_subjects = np.array(os.listdir(rhino_root + '/scratch/tphan/joint_classifier/FR1/'))
subject = all_subjects[index]
print subject
subject = 'R1001P'

subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
dataset = joblib.load(subject_dir)
sessions = np.unique(dataset['session'])

dataset_enc = select_phase(dataset)
sigma_noise = 0.05

dataset['X'] = normalize_sessions(dataset['X'], dataset['session'])
dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])
result_current = run_loso_xval(dataset_enc, classifier_name = 'current', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)


dataset_auto = get_session_data(subject, rhino_root)
dataset_auto['X'] = normalize_sessions(dataset_auto['X'], dataset_auto['session'])
#dataset_enc = select_phase(dataset)
#dataset_enc['X'] = scale_sessions(dataset_enc['X'], dataset_enc['session'], dataset_enc['X'], dataset_enc['session'])
#dataset_auto['X'] = scale_sessions(dataset_auto['X'], dataset_auto['session'], dataset_auto['X'], dataset_auto['session'])


print dataset_enc['X'].min()
print dataset_enc['X'].max()


print dataset_auto['X'].min()

print dataset_auto['X'].max()

#dataset_auto = dataset_enc
corruption_level = 1.0

probs_all=[]
label_all =[]

# pre-training
x_encoded_vec = []
y_encoded_vec = []


if len(sessions) > 1:
    for sess in sessions:
        print "session = ", sess

        session = dataset_auto['session']
        session_mask = dataset_auto['session'] != sess
        train_data = dataset_auto['X'][session_mask]
        test_data = dataset_auto['X'][~session_mask]

        #train_weights = np.ones(len(train_label))
        session_enc = dataset_enc['session']
        insample_list = dataset_enc['list']
        session_mask_enc = dataset_enc['session'] != sess
        train_data_enc = dataset_enc['X'][session_mask_enc]
        train_data_label = dataset_enc['y'][session_mask_enc]
        test_data_enc = dataset_enc['X'][~session_mask_enc]
        train_data_label_cat = np_utils.to_categorical(train_data_label,2)

        n_features = train_data.shape[1]
        print("number of features = {0:4d}").format(n_features)
        print("number of events = {0:4d}").format(train_data.shape[0])

        autoencoder_model = create_model(dropout_rate=0.4, layers = [n_features,512], activation= 'sigmoid', penalty = 0.0e-5)
        if n_features > 1000:
            callback = [EarlyStopping(monitor = 'val_loss', patience = 50, min_delta=0.01, mode = 'auto')]
        else:
            callback = [EarlyStopping(monitor = 'val_loss', patience = 25, min_delta=0.01, mode = 'auto')]

        validate_percent = 0.10

        if denoising:
            print "running denoising autoencoder ... "
            train_data_noisy = train_data + corruption_level*np.random.normal(0,1.0, size = train_data.shape)
            validate_data = train_data_enc
            #validate_data_noisy = train_data_enc + corruption_level*np.random.normal(0,1.0, size = train_data_enc.shape)
            autoencoder_model.fit(train_data_noisy,train_data, epochs = 100, batch_size = 24, shuffle = True, validation_data = (validate_data,validate_data),
                            callbacks = callback)

        layer_name = 'encoder'

        intermediate_layer_model = Model(inputs = autoencoder_model.input, outputs=autoencoder_model.get_layer(layer_name).output)

        model = Model(inputs = autoencoder_model.input, outputs=autoencoder_model.get_layer(layer_name).get_output_at(0))
        train_encoded = model.predict(train_data)

        #intermediate_layer_model = Model(inputs = model.input, outputs=model.get_layer(layer_name).get_output_at(1))

        X_recons = autoencoder_model.predict(train_data)
        print "max encoder", train_encoded.max()
        print "min encoder", train_encoded.min()
        print "min recons", X_recons.min()
        print "min recons", X_recons.max()
        for layer in autoencoder_model.layers:
            print layer


        #encoder = layers.concatenate(autoencoder_model.layers[:2])
        model = models.Sequential()
        for layer in autoencoder_model.layers[:2]:
            model.add(layer)

        model.add(layers.Dense(2,  activation = 'softmax', kernel_regularizer=regularizers.l2(7.2e-7)))
        model.compile(optimizer =  'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


        test_data_label_enc = dataset_enc['y'][~session_mask_enc]
        training_encoded = intermediate_layer_model.predict(train_data_enc)

        print training_encoded.shape

        feature_std =  training_encoded.std(axis = 0)
        print len(feature_std)

        indices = np.where(feature_std > 0.00001)[0]
        print feature_std[indices]
        print indices
        training_encoded = training_encoded[:,indices]
        print training_encoded.shape

        # probs = model.predict(test_data_enc)[:,1]
        test_encoded = intermediate_layer_model.predict(test_data_enc)
        test_encoded = test_encoded[:,indices]

        ind_params = {'class_weight':'balanced', 'solver':'saga'}
        best_params = opt_params(training_encoded, train_data_label, session_enc[session_mask_enc], insample_list, 'L2', ind_params, 'rand', 'short', False) # tuning parameters

        #ind_params.update(best_params)
        print training_encoded.min()
        print training_encoded.max()

        classifier = LogisticRegression(**ind_params)
        classifier.fit(training_encoded, train_data_label)


        probs = classifier.predict_proba(test_encoded)[:,1]
        probs_all.append(probs)
        label_all.append(test_data_label_enc)



    label_all = np.concatenate(label_all)
    probs_all = np.concatenate(probs_all)
    auc_auto = sklearn.metrics.roc_auc_score(label_all,probs_all)
    print auc_auto
    print result_current
