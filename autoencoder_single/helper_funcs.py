import os
from ptsa.data.readers import JsonIndexReader
from ptsa.data.readers import TalReader
from ptsa.data.readers import EEGReader
from ptsa.data.readers import H5RawReader
import numpy as np
import h5py
from ptsa.data import TimeSeriesX

import keras
from keras import optimizers
from keras import regularizers
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras import models, layers
from sklearn.externals import joblib
def create_paths(subject):
    pairs_path = os.path.join('/Volumes/RHINO/'
    'protocols', 'r1', 'subjects', subject,
    'localizations', str(0),
    'montages', str(0),
    'neuroradiology', 'current_processed', 'pairs.json')

    paths = FilePaths(
        root='/Volumes/RHINO/',
        pairs=pairs_path,
        dest='/scratch/tphan/FR1/reports/',
        data_db='/scratch/tphan/FR1/'
    )

    return (pairs_path, paths)


def get_bipolar_pairs(subject, experiment, sessions, paths):

    stim_params = 'None'
    ec_pairs = get_pairs(subject, experiment,sessions, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    result = build_montage_metadata_table(subject, experiment,sessions, ec_pairs.compute(), paths.root)
    bipolar_pairs = result[['channel_1', 'channel_2']]
    bipolar_pairs = bipolar_pairs.astype(int)

    return result



def read_session(subject, root, experiment, rhino_root):
    jr = JsonIndexReader(rhino_root + '/protocols/r1.json')  # Build JSON reader
    pairs_path = jr.get_value('pairs', subject=subject, experiment=experiment)
    tal_reader = TalReader(filename=pairs_path)
    monopolar_channels = tal_reader.get_monopolar_channels()
    bipolar_channels = tal_reader.get_bipolar_pairs()

    jr = JsonIndexReader(rhino_root + '/protocols/r1.json')  # Build JSON reader
    pairs_path = jr.get_value('pairs', subject=subject, experiment=experiment)
    tal_reader = TalReader(filename=pairs_path)
    monopolar_channels = tal_reader.get_monopolar_channels()
    p_reader = ParamsReader(dataroot=root)
    params = p_reader.read()
    samplerate = params['samplerate']
    print samplerate
    try:
        raw_eeg = EEGReader(session_dataroot=root, channels = monopolar_channels)
        raw_eeg = raw_eeg.read()
        print raw_eeg.shape

    except:
        with h5py.File(root, 'r') as hfile:
            raw_eeg = H5RawReader.H5RawReader.read_h5file(hfile, monopolar_channels,[0])[0]
            print raw_eeg.shape
            raw_eeg = TimeSeriesX.TimeSeriesX(raw_eeg,
                                  dims=['channels', 'start_offsets', 'time'],
                                  coords={
                                      'channels': monopolar_channels,
                                      'start_offsets': [0],
                                      'time': np.arange(raw_eeg.shape[2])*1000/samplerate,
                                      'offsets': ('time',np.arange(raw_eeg.shape[2])),
                                      'samplerate': samplerate})
    return raw_eeg



def array_mean(array, axis = 0, interval_length = 500):

    dim = array.shape
    n_intervals = np.int(dim[axis]/interval_length)
    indices = np.array_split(np.arange(dim[axis]), n_intervals)

    final_shape = [dim[x] for x in np.arange(len(dim)) if x != axis] + [n_intervals]
    result = np.zeros(shape = final_shape)

    for i in np.arange(len(indices)):
        result[:,:,i] = np.nanmean(array[:,:,indices[i]], axis = 2)


    return result

import keras.backend as K
def mse(y_true, y_pred):
    return K.mean(K.pow((y_true-y_pred),2))

def create_model(dropout_rate = 0.4, layers = [200,100], penalty = 0.0, regularizer = regularizers.l2, activation = 'relu'):
    print activation
    print regularizer
    if len(layers) < 3:
        autoencoder = create_model_one_layer(dropout_rate,layers, penalty)
    else:
        print layers
        n_features = layers[0]
        input_eeg = Input(shape = (n_features,))
        hidden_units = layers[1:]

        encoded = Dense(hidden_units[0], activation= activation,kernel_regularizer=regularizer(penalty))(input_eeg)
        #encoded = Dropout(rate = dropout_rate )(encoded)
        for i,hidden_unit in enumerate(hidden_units[1:]):

            if i == (len(hidden_units[1:])-1):
                encoded = Dense(hidden_unit, activation= activation, kernel_regularizer=regularizer(penalty), name = 'encoder')(encoded)
            else:
                encoded = Dense(hidden_unit, activation= activation, kernel_regularizer=regularizer(penalty))(encoded)

            #encoded = Dropout(rate = dropout_rate )(encoded)
        hidden_units_reversed = hidden_units[::-1]
        print hidden_units_reversed[1]

        decoded = Dense(hidden_units_reversed[1], activation= activation,kernel_regularizer=regularizer(penalty))(encoded)
        #decoded = Dropout(rate = dropout_rate )(decoded)


        for hidden_unit in hidden_units_reversed[2:]:
            print hidden_unit
            decoded = Dense(hidden_unit, activation = activation, kernel_regularizer= regularizer(penalty))(decoded)
            #decoded = Dropout(rate = dropout_rate)(decoded)



        decoded = Dense(n_features, name = 'decoder', activation = activation)(decoded)
        autoencoder = Model(input_eeg, decoded)
        rmsprop = optimizers.RMSprop(lr= 5.0e-4, rho = 0.9)

        autoencoder.compile(optimizer = rmsprop, loss = 'mean_squared_error', metrics = [mse])
    return autoencoder

def create_model_one_layer(dropout_rate = 0.4, layers = [784,32], penalty = 0.0, activation = 'relu'):
    print layers
    model = Sequential()

    n_features = layers[0]
    input_eeg = Input(shape = (n_features,))
    hidden_units = layers[1:]
    model.add(Dense(hidden_units[0], input_dim = n_features, kernel_regularizer= regularizers.l2(penalty), activation = 'relu', name = 'encoder'))
    #model.add(BatchNormalization())

    model.add(Dense(layers[0], name = 'decoded'))
    #model.add(BatchNormalization())
    rmsprop = optimizers.RMSprop(lr= 1.0e-4, rho = 0.9)

    model.compile(optimizer = rmsprop, loss = 'mean_squared_error', metrics = [mse])
    return model


# stacked autoencoder
def stacked_autoencoder(X, layers, validate_rate = 0.05, penalty = 0.0, corruption_level = 0.5):
    n_features = layers[0]
    hidden_units = layers[1:]
    input = X
    print layers
    intermediate_layer_models = []

    print hidden_units[0:-1]
    for i, hidden_unit in enumerate(hidden_units):
        if i == 0:
            layers = [n_features] + [hidden_unit]
            autoencoder_model = create_model(dropout_rate=0.4, layers = layers, penalty = penalty)
        else:
            layers = [hidden_units[i-1]] + [hidden_unit]
            autoencoder_model = create_model(dropout_rate=0.4, layers = layers, penalty = 0.0)


        if n_features > 1000:
            callback = [EarlyStopping(monitor = 'val_loss', patience = 100, min_delta=0.005, mode = 'auto')]
        else:
            callback = [EarlyStopping(monitor = 'val_loss', patience = 100, min_delta=0.005, mode = 'auto')]

        result = train_test_split(input, test_size = validate_rate, shuffle = False)
        train_data_sub = result[0]
        validate_data = result[1]
        train_data_noisy = train_data_sub + corruption_level*np.random.normal(0,1.0, size = train_data_sub.shape)
        validate_data_noisy = validate_data + corruption_level*np.random.normal(0,1.0, size = validate_data.shape)
        autoencoder_model.fit(train_data_noisy,train_data_sub, epochs = 300, batch_size = 24, shuffle = True, validation_data = (validate_data_noisy,validate_data))

        layer_name = 'encoder'
        intermediate_layer_model = Model(inputs = autoencoder_model.input, outputs=autoencoder_model.get_layer(layer_name).output)
        intermediate_layer_models.append(intermediate_layer_model)
        input =  intermediate_layer_model.predict(input)
        print "encoder shape = ", input.shape

    return intermediate_layer_models


def train_auto_encoder(X, hidden_layers = [512], corruption_level = 0.1, denoising = True, activation = 'relu', n_epochs = 200, autoencoder_model = None, validate_data = None,
penalty = 0.0 ):
        # build autoencoder for all other subjects
        n_features_first = X.shape[1]
        layers = [n_features_first] + hidden_layers
        print layers
        if autoencoder_model is None:
            autoencoder_model = create_model(dropout_rate=0.4, layers = layers, activation= activation, penalty = penalty)
        if n_features_first > 1000:
            callback = [EarlyStopping(monitor = 'val_loss', patience = 50, min_delta=0.005, mode = 'auto')]
        else:
            callback = [EarlyStopping(monitor = 'val_loss', patience = 35, min_delta=0.005, mode = 'auto')]

        validate_percent = 0.10
        if denoising:
            print "running denoising autoencoder ... "

            if validate_data is None:
                result = train_test_split(X, test_size = validate_percent)
                train_data_sub = result[0]
                validate_data = result[1]
                train_data_noisy = train_data_sub + corruption_level*np.random.normal(0,1.0, size = train_data_sub.shape)
                validate_data_noisy = validate_data + corruption_level*np.random.normal(0,1.0, size = validate_data.shape)
                autoencoder_model.fit(train_data_noisy,train_data_sub, epochs = n_epochs, batch_size = 24, shuffle = True, validation_data = (validate_data_noisy,validate_data),
                                callbacks = callback)
            else:
                train_data_noisy = X + corruption_level*np.random.normal(0,1.0, size = X.shape)

                validate_data_noisy = validate_data + corruption_level*np.random.normal(0,1.0, size = validate_data.shape)

                autoencoder_model.fit(train_data_noisy,X, epochs = n_epochs, batch_size = 24, shuffle = True, validation_data = (validate_data_noisy,validate_data),
                                callbacks = callback)

        return autoencoder_model


def build_prior_model(subject_current, rhino_root = '', threshold = 0.6, n_epochs = 200):
    print rhino_root
    all_subjects = np.array(os.listdir(rhino_root + '/scratch/tphan/joint_classifier/FR1/'))
    save_dir_model_prior = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject_current + '/prior_model_' + str(threshold) + '.h5'
    save_dir_model_auto = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject_current + '/autoencoder_model_' + str(threshold) + '.h5'

    X_encoder = []
    session = []
    subject_list = []
    y_list = []
    sample_weights = []
    noise_opt = []
    for subject in all_subjects:
        print subject
        print subject_current
        if subject != subject_current:
            print "True"
            try:
                subject_dir_save = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/autoencoder_keras_cv_session_greedy_noise_relu.pkl'
                data_subject = joblib.load(subject_dir_save)
                subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
                dataset = joblib.load(subject_dir)
                dataset_enc = select_phase(dataset)
                subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/current.pkl'
                result_current = joblib.load(subject_dir)
                print data_subject['auc']
                print result_current['comb']
                if result_current['comb'] > threshold:
                    assert data_subject['encoder'].shape[0] == len(dataset_enc['session'])
                    X_encoder.append(data_subject['encoder'])
                    session.append(dataset_enc['session'])
                    subject_list.append(np.repeat(subject, len(dataset['session'])))
                    sample_weights.append(get_sample_weights_fr(dataset_enc['y']))
                    y_list.append(dataset_enc['y'])
                    noise_opt.append(data_subject['noise_opt'])
                    print data_subject['noise_opt']
            except:
                print "data does not exist for " + subject

    print noise_opt
    print len(subject_list)
    noise_opt = np.array(noise_opt)

    sample_weights = np.concatenate(sample_weights)
    X_encoder = np.concatenate(X_encoder)
    session = np.concatenate(session)
    subject_list  = np.concatenate(subject_list)
    y_list = np.concatenate(y_list)
    print len(y_list)
    print X_encoder.shape

    autoencoder_model = train_auto_encoder(X_encoder, hidden_layers=[256], n_epochs = n_epochs)

    # fine-tuning using recall events
    print autoencoder_model.layers
    model = models.Sequential()
    for layer in autoencoder_model.layers[:2]:
        print layer
        model.add(layer)

    n_features = X_encoder.shape[1]
    if n_features > 1000:
        callback = [EarlyStopping(monitor = 'val_loss', patience = 100, min_delta=0.01, mode = 'auto')]
    else:
        callback = [EarlyStopping(monitor = 'val_loss', patience = 50, min_delta=0.01, mode = 'auto')]

    model.add(layers.Dense(2,  activation = 'softmax', kernel_regularizer=regularizers.l2(7.2e-4)))
    model.compile(optimizer =  'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print model.layers
    layer_name = 'encoder'
    train_data_label_cat = np_utils.to_categorical(y_list,2)
    X_train, X_test, y_train, y_test = train_test_split(X_encoder, train_data_label_cat, test_size = 0.10, stratify = y_list)
    print "fine tuning ..."
    model.fit(X_train, y_train, epochs = n_epochs, batch_size= 24, validation_data = (X_test,y_test), callbacks = callback )

    save_dir_model = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject_current + '/prior_model_' + str(threshold) + '.h5'
    model.save(save_dir_model)

    save_dir_model = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject_current + '/autoencoder_model_' + str(threshold) + '.h5'
    autoencoder_model.save(save_dir_model)
    return autoencoder_model, model


def get_session_data(subject, rhino_root = ''):

    subject_dir_auto = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset_long_50.0.pkl'
    dataset_auto = joblib.load(subject_dir_auto)
    print dataset_auto[0]['pow'].shape
    try:
        sessions = [x['session'] for x in dataset_auto]
    except:
        print "not in dict form "

    try:
        sessions = [x.session.values for x in dataset_auto]
    except:
        print "not in Time Series X"

    X_sess_vec = []
    sess_vec = []
    try:

        for i,session in enumerate(sessions):
            X_sess = dataset_auto[i]
            session_label = dataset_auto[i].session.values
            n_freqs, n_pairs, n_events = X_sess.shape
            X_sess_reshape = X_sess.values.T.reshape((n_events,-1))
            X_sess_vec.append(X_sess_reshape)
            sess_vec.append(np.repeat(session, n_events))
    except:
        print "not in Timeseries X"

    try:
        for i,session in enumerate(sessions):

            X_sess = dataset_auto[i]['pow']
            n_freqs, n_pairs, n_events = X_sess.shape
            X_sess_reshape = X_sess.T.reshape((n_events,-1))
            X_sess_vec.append(X_sess_reshape)
            sess_vec.append(np.repeat(session, n_events))
    except:
        print "not in dict format"

    dataset_auto = {'X': np.concatenate(X_sess_vec), 'session': np.concatenate(sess_vec)}

    return dataset_auto

