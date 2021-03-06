# adversarial autoencoder to combine datasets across subjects

import matplotlib
matplotlib.use('Agg')
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
#sys.path.append(os.getcwd() + '/autoencoder_single')
sys.path.append(os.getcwd())
from classifier import*
from helper_funcs import*


def build_encoder(input_dim, latent_dim, N = 64, n_classes = 2, penalty = 0, n_hidden = 1):
    input = Input(shape = (input_dim,))
    h = Dense(N, kernel_regularizer=regularizers.l2(penalty))(input)
    h = LeakyReLU(alpha=0.2)(h)
    h = BatchNormalization()(h)
        h = Dropout(0.4)(h)

    for i in range(n_hidden):
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization()(h)
        h = Dropout(0.4)(h)
    # latent code model
    h = Dense(latent_dim)(h)
    h = LeakyReLU(alpha=0.2)(h)
    model_encoder = Model(input, h)
    model_encoder.summary()
    return model_encoder


# fix this later
def build_decoder(output_dim, code_dim, N = 64, activation = 'sigmoid', penalty = 0, n_hidden =1 ):
    model = Sequential()
    model.add(Dense(N, input_dim = code_dim, kernel_regularizer=regularizers.l2(penalty)))

    for i in range(n_hidden):
        model.add(LeakyReLU(alpha = 0.2))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

    model.add(Dense(output_dim, activation = activation)) # normalize data between -1 and 1
    model.summary()
    z = Input(shape = (code_dim,))
    output = model(z)
    model_decoder = Model(z,output)

    return model_decoder


# discriminate between code and prior distribution
def build_discriminator_gauss(latent_dim, N = 64, n_hidden = 1):

    model = Sequential()
    model.add(Dense(N, input_dim=latent_dim))

    for i in range(n_hidden):
        model.add(LeakyReLU(alpha =0.2))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    encoded_repr = Input(shape = (latent_dim,))
    validity = model(encoded_repr)

    model = Model(encoded_repr, validity)
    model.summary()

    return model


# discriminate category
def build_discriminator_cat(n_classes, N = 64, n_hidden =2):

    model = Sequential()
    model.add(Dense(N, input_dim=n_classes))

    for i in range(n_hidden):
        model.add(LeakyReLU(alpha =0.2))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()

    encoded_repr = Input(shape = (n_classes,))
    validity = model(encoded_repr)

    model = Model(encoded_repr, validity)
    model.summary()
    return model


def soft_max_classifier(latent_dim):
    print latent_dim
    print "type of latent_dim", type(latent_dim)
    model = Sequential()

    model.add(Dense(1, input_dim= latent_dim))
    model.add(Activation('sigmoid'))
    #model.summary()

    encoded_repr = Input(shape = (latent_dim,))
    y_tilde = model(encoded_repr)
    model = Model(encoded_repr, y_tilde)
    model.summary()

    return model


def sample_images(epoch, latent_dim, decoder, X_train):

    n = 10  # how many digits we will display
    fig, axes = plt.subplots(2,5)
    z = np.random.normal(size=(10, latent_dim))
    real_cat_dist = np.random.randint(low=0, high=10, size=10)
    real_cat_dist = np.eye(10)[real_cat_dist]
    z = np.concatenate([real_cat_dist, z], axis = 1)
    gen_X_batch = decoder.predict(z)
    gen_X_batch = 0.5 * gen_X_batch + 0.5

    for i in range(n):
        i2 = i%5
        i1 = i/5
        axes[i1,i2].imshow(gen_X_batch[i].reshape(28, 28))
        axes[i1,i2].get_xaxis().set_visible(False)
        axes[i1,i2].get_yaxis().set_visible(False)

    fig.savefig("/Volumes/RHINO/home2/tungphan/superAutoencoder/autoencoder_single/images/mnist_%d.png" % epoch)

    fig, axes = plt.subplots(2,5)


    gen_X_batch = X_train
    gen_X_batch = 0.5 * gen_X_batch + 0.5

    print axes.shape

    for i in range(n):
        i2 = i%5
        i1 = i/5

        # display original
        # axes[i1,i2] = plt.subplot(2, n, i + 1)
        # plt.imshow(X_train[i,:,:,0])
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # display reconstruction

        axes[i1,i2].imshow(gen_X_batch[i].reshape(28, 28))
        axes[i1,i2].get_xaxis().set_visible(False)
        axes[i1,i2].get_yaxis().set_visible(False)
    fig.savefig("/Volumes/RHINO/home2/tungphan/superAutoencoder/autoencoder_single/images/mnist_%d_real.png" % epoch)


def load_dataset(index, rhino_root):

    all_subjects = np.array(os.listdir(rhino_root + '/scratch/tphan/joint_classifier/FR1/'))
    subject = all_subjects[index]
    #subject = 'R1065J'
    print subject

    subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
    dataset = joblib.load(subject_dir)
    dataset_auto = get_session_data(subject, rhino_root)
    dataset_enc = select_phase(dataset)

    return dataset_enc, dataset_auto



if __name__ == '__main__':


    args = sys.argv
    print args
    index = int(args[1])
    penalty = pow(10,-int(args[2]))
    n_hidden = int(args[3])


    latent_dim = 10
    n_classes = 2

    rhino_root = '/Volumes/RHINO'
    dataset_enc, dataset_auto = load_dataset(index, rhino_root)
    sessions = np.unique(dataset_enc['session'])

    dataset_enc_temp = dataset_enc
    dataset_enc_temp['X'] = normalize_sessions(dataset_enc_temp['X'], dataset_enc_temp['session'])
    result_current = run_loso_xval(dataset_enc_temp, classifier_name = 'current', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)

    print result_current

    dataset_enc['X'] = scale_sessions(dataset_enc['X'], dataset_enc['session'], dataset_enc['X'], dataset_enc['session'])
    dataset_auto['X'] = scale_sessions(dataset_auto['X'], dataset_auto['session'], dataset_auto['X'], dataset_auto['session'])

    # training
    batch_size = 12

    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    sessions = np.unique(dataset_enc['session'])

    prob_all = []
    y_all = []


    for sess in sessions:

        all_sess_mask = dataset_auto['session'] == sess
        enc_sess_mask = dataset_enc['session'] == sess
        X_train_unlabeled = dataset_auto['X'][~all_sess_mask]

        print "number of samples = ", X_train_unlabeled.shape[0]
        X_train_labeled = dataset_enc['X'][~enc_sess_mask]
        y_train_labeled = dataset_enc['y'][~enc_sess_mask]
        X_test_labeled = dataset_enc['X'][enc_sess_mask]
        y_test_labeled = dataset_enc['y'][enc_sess_mask]

        # adversarial ground truth
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        #valid = np.random.uniform(0.7,1.2,size =(batch_size,1))
        #fake = np.random.uniform(0.0,0.3,size =(batch_size,1))

        input_dim = dataset_enc['X'].shape[1]
        output_dim = dataset_enc['X'].shape[1]
        discriminator_gauss= build_discriminator_gauss(latent_dim, n_hidden = n_hidden)
        optimizer = Adam(0.0002, 0.5)
        corruption_level = 0.05

        discriminator_gauss.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        encoder = build_encoder(input_dim, latent_dim, n_classes = n_classes, n_hidden = n_hidden)
        decoder = build_decoder(output_dim, latent_dim , n_hidden = n_hidden)

        input = Input(shape = (input_dim,))
        z = encoder(input)

        reconstructed_input = decoder(z)
        discriminator_gauss.trainable = False
        validity_gauss = discriminator_gauss(z)
        adversarial_gauss = Model(input, outputs = [reconstructed_input, validity_gauss])
        adversarial_gauss.compile(loss = ['mse', 'binary_crossentropy'], loss_weights=[0.999,0.001], optimizer = optimizer)


        #discriminator_gauss.trainable = True

        L2= soft_max_classifier(latent_dim)
        y_cat = L2(encoder(input))

        encoder.trainable = False

        supervised_classifier = Model(input = input, output = y_cat)
        supervised_classifier.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        print "supervised classifier summary", supervised_classifier.summary()

        #discriminator_gauss.trainable = True

        n_epochs = 2000
        sample_interval = 200
        n_iter = n_epochs*X_train_labeled.shape[0]/batch_size

        sample_weights, pos_weight, neg_weight = get_sample_weights_fr(y_train_labeled)
        print pos_weight, neg_weight
        class_weight = {0:neg_weight, 1:pos_weight}

        print("number of iterations {}".format(n_iter))

        for epoch in range(n_iter):

            idx_unlabeled = np.random.randint(0, X_train_unlabeled.shape[0], batch_size)
            X_batch_unlabeled = X_train_unlabeled[idx_unlabeled]

            idx_labeled = np.random.randint(0, X_train_labeled.shape[0], batch_size)
            X_batch_labeled = X_train_labeled[idx_labeled]

            y_batch_labeled = y_train_labeled[idx_labeled]
            y_batch_labeled = np.eye(n_classes)[y_batch_labeled]

            X_batch_unlabeled_noise = X_batch_unlabeled  + np.random.normal(0,1,size = X_batch_unlabeled.shape)*corruption_level

            latent_fake = encoder.predict(X_batch_unlabeled_noise)
            latent_real = np.random.normal(size = (batch_size, latent_dim))*5

            real_cat_dist = np.random.randint(low=0, high=n_classes, size=batch_size)
            real_cat_dist = np.eye(n_classes)[real_cat_dist]

            # Train Gaussian AN
            d_loss_gauss_real = discriminator_gauss.train_on_batch(latent_real, valid)
            d_loss_gauss_fake = discriminator_gauss.train_on_batch(latent_fake, fake)
            d_loss_gauss = 0.5*np.add(d_loss_gauss_real, d_loss_gauss_fake)
            g_loss_gauss = adversarial_gauss.train_on_batch(X_batch_unlabeled_noise, [X_batch_unlabeled, valid])

            # Train Cat AN
            discriminator_gauss.predict(latent_real)


           # if epoch%10 == 0: # update once in awhile

            supervised_loss = supervised_classifier.train_on_batch(X_batch_labeled, y_batch_labeled, class_weight = class_weight)

            roc_in = roc_auc_score(y_train_labeled, supervised_classifier.predict(X_train_labeled)[:,1])
            y_pred = supervised_classifier.predict(X_test_labeled)[:,1]
            roc_val = roc_auc_score(y_test_labeled, y_pred)

            if epoch%sample_interval == 0:
                print ("%d [D loss Gauss: %f, acc: %.2f%%] [G loss Gauss: %f, mse: %f]" % (epoch, d_loss_gauss[0], 100*d_loss_gauss[1], g_loss_gauss[0], g_loss_gauss[1]))
                print ("%d [Supervised Loss : %f, acc: %.2f%%] " % (epoch, supervised_loss[0], 100*supervised_loss[1]))
                print ("%d [Validation Loss : %f, acc: %.2f%%, auc_val: %f, auc_in: %f] " % (epoch, test_accuracy[0], 100*test_accuracy[1], roc_val, roc_in))
                print ("_____________________________________________")

        training_all = encoder.predict(X_train_unlabeled)
        training_labeled = encoder.predict(X_train_labeled)

        prob_all.append(y_pred)
        y_all.append(y_test_labeled)
    auc = roc_auc_score(np.concatenate(y_all), np.concatenate(prob_all))
    print( 'AUC current = {}, AUC auto = {} '.format(result_current['comb'], auc))




# plot distributions
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt
fig, ax = plt.subplots(8,1)
sns.set(color_codes= True)
for i in np.arange(8):
    sns.distplot(training_all[:,i], ax = ax[i], color = 'green')
    sns.distplot(np.random.normal(size = 1000), ax = ax[i], color = 'red')

fig.savefig('test.pdf')



# plot distributions
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt
fig, ax = plt.subplots(8,1, sharex= True)
sns.set(color_codes= True)
for i in np.arange(8):
    sns.distplot(training_labeled[:,i], ax = ax[i], color = 'green')
    sns.distplot(np.random.normal(size = 1000), ax = ax[i], color = 'red')

fig.savefig('test_labeled.pdf')
