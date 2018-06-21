import numpy as np
import ptsa
#import ramutils
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
from adversarial_autoencoder import*
import helper_funcs
from helper_funcs import*
import classifier
from classifier import*
from keras.utils import to_categorical


from keras.callbacks import TensorBoard
from time import time
import tensorflow


args = sys.argv
print args
index = int(args[1])
denoising = args[2]

rhino_root = '/Volumes/RHINO'
all_subjects = np.array(os.listdir(rhino_root + '/scratch/tphan/joint_classifier/FR1/'))
subject = all_subjects[index]
print subject
#subject = 'R1032D'

subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
dataset = joblib.load(subject_dir)
sessions = np.unique(dataset['session'])

dataset_enc = select_phase(dataset)
sigma_noise = 0.1

dataset['X'] = normalize_sessions(dataset['X'], dataset['session'])
dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])
result_current = run_loso_xval(dataset_enc, classifier_name = 'current', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)

print "current result", result_current

dataset_auto = get_session_data(subject, rhino_root)
dataset_auto['X'] = normalize_sessions(dataset_auto['X'], dataset_auto['session'])
dataset_enc = select_phase(dataset)


print dataset_enc['X'].min(), dataset_enc['X'].max()
print dataset_auto['X'].min(), dataset_auto['X'].max()

dataset_enc['X'] = scale_sessions(dataset_enc['X'], dataset_enc['session'], dataset_enc['X'], dataset_enc['session'])
dataset_auto['X'] = scale_sessions(dataset_auto['X'], dataset_auto['session'], dataset_auto['X'], dataset_auto['session'])


print dataset_enc['X'].min(), dataset_enc['X'].max()
print dataset_auto['X'].min(), dataset_auto['X'].max()

#dataset_auto = dataset_enc
corruption_level = 0.01

X_train = np.concatenate([dataset_auto['X'], dataset_enc['X']], axis = 0)

input_dim = dataset_enc['X'].shape[1]
latent_dim = 10
output_dim = input_dim

probs_all=[]
label_all =[]



if len(sessions) > 1:
    for sess in sessions:

        session = dataset_auto['session']
        session_mask = dataset_auto['session'] != sess
        train_data = dataset_auto['X'][session_mask]
        test_data = dataset_auto['X'][~session_mask]


        session_enc = dataset_enc['session']
        insample_list = dataset_enc['list']
        session_mask_enc = dataset_enc['session'] != sess
        train_data_enc = dataset_enc['X'][session_mask_enc]
        train_data_label = dataset_enc['y'][session_mask_enc]
        test_data_enc = dataset_enc['X'][~session_mask_enc]
        train_data_label_cat = np_utils.to_categorical(train_data_label,2)
        test_data_label_enc = dataset_enc['y'][~session_mask_enc]


        X_train = np.concatenate([train_data, train_data_enc], axis = 0)
        print "number of training samples", X_train.shape[0]
        print "pas here"


        discriminator= build_discriminator_semi(latent_dim)
        optimizer = Adam(0.0002, 0.5)
        discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        encoder = build_encoder(input_dim, latent_dim)
        decoder = build_decoder(output_dim, latent_dim)


        input_noise = Input(shape = (input_dim,))
        encoded_repr = encoder(input_noise)
        reconstructed_input = decoder(encoded_repr)

        discriminator.trainable = False
        validity = discriminator(encoded_repr)


        adversarial_autoencoder = Model(input_noise, outputs = [reconstructed_input, validity])
        adversarial_autoencoder.compile(loss = ['mse', 'binary_crossentropy'], loss_weights=[0.99,0.01], optimizer = optimizer)


        # classifier_tune = Model(input_noise, y_tilde)
        # classifier_tune.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        # build a classifier based on code

        # training
        batch_size = 12

        # adversarial ground truth


        epochs = 20000
        sample_interval = 200

        sample_weights, pos_weight, neg_weight = get_sample_weights_fr(train_data_label)
        print pos_weight, neg_weight
        class_weight = {0:neg_weight, 1:pos_weight}

        train_data_label = to_categorical(train_data_label)




        for epoch in range(epochs):

            valid = np.zeros((batch_size,1))
            fake = np.ones((batch_size,1))
            idx = np.random.randint(0, train_data_enc.shape[0], batch_size)
            imgs_enc = train_data_enc[idx]
            imgs_enc_noise = imgs_enc + sigma_noise*np.random.normal(size = imgs_enc.shape)
            y_batch = train_data_label[idx]

            valid = np.concatenate([y_batch, valid], axis = 1)
            fake = np.concatenate([np.zeros(shape = y_batch.shape), fake], axis = 1)
            latent_fake = encoder.predict(imgs_enc_noise)
            latent_real = np.random.normal(size = (batch_size, latent_dim))

            # train discriminator
            d_loss_real = discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

            # train generator
            g_loss = adversarial_autoencoder.train_on_batch(imgs_enc_noise, [imgs_enc, valid])

            c_loss = [0,0]
            # train classifier
            if epoch%sample_interval == 0:
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f] [C loss: %f, acc: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], c_loss[0], c_loss[1]))
                #sample_images(epoch, latent_dim, decoder, imgs)
                #discriminator_grads = get_gradients(discriminator)
                #adversarial_autoencoder_grads = get_gradients(adversarial_autoencoder)
                #print discriminator_grads
                #print adversarial_autoencoder_grads



        test_code = encoder.predict(test_data_enc)
        probs = discriminator.predict(test_code)[:,:2]
        probs = probs[:,1]/probs.sum(axis = 1)



        probs_all.append(probs)
        label_all.append(test_data_label_enc)



    label_all = np.concatenate(label_all)
    probs_all = np.concatenate(probs_all)
    auc_auto = sklearn.metrics.roc_auc_score(label_all,probs_all)
    print "auto auc", auc_auto
    print "current auc", result_current

    # save_dir = rhino_root + '/scratch/tphan/superautoencoder/' + subject + '/'
    #
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # result_all = collections.OrderedDict()
    # result_all['aae'] = auc_auto
    # result_all['current'] = result_current
    # #result_all['L2'] = result_L2
    # joblib.dump(result_all, save_dir + 'aae_result.pkl')
    # print result_all


