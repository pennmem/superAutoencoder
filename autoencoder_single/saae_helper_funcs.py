# adversarial autoencoder to combine datasets across subjects

import matplotlib
matplotlib.use('Agg')
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Dropout, concatenate, Softmax
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


def build_encoder(input_dim, latent_dim, N = 1000, n_classes = 2):
    input = Input(shape = (input_dim,))
    h = Dense(N)(input)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dropout(0.4)(h)

    h = Dense(N)(input)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dropout(0.4)(h)



    # categorical model
    y_cat = Dense(n_classes, activation= 'softmax')(h)  # softmax classifier

    # latent code model
    h = Dense(latent_dim)(h)
    h = LeakyReLU(alpha=0.2)(h)
    model_encoder = Model(input, h)
    model_cat = Model(input, y_cat)
    model_encoder.summary()
    return model_encoder, model_cat


# fix this later
def build_decoder(output_dim, code_dim, N = 128):
    model = Sequential()
    model.add(Dense(N, input_dim = code_dim))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))


    model.add(Dense(N))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))


    model.add(Dense(output_dim, activation = 'tanh')) # normalize data between -1 and 1
    model.summary()
    z = Input(shape = (code_dim,))
    output = model(z)
    model_decoder = Model(z,output)

    return model_decoder


# discriminate between code and prior distribution
def build_discriminator_gauss(latent_dim, N = 128):

    model = Sequential()
    model.add(Dense(N, input_dim=latent_dim))
    model.add(LeakyReLU(alpha =0.2))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    encoded_repr = Input(shape = (latent_dim,))
    validity = model(encoded_repr)

    model = Model(encoded_repr, validity)
    model.summary()

    return model


# discriminate category
def build_discriminator_cat(n_classes, N = 128):

    model = Sequential()
    model.add(Dense(N, input_dim=n_classes))
    model.add(LeakyReLU(alpha =0.2))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()

    encoded_repr = Input(shape = (n_classes,))
    validity = model(encoded_repr)

    model = Model(encoded_repr, validity)
    model.summary()
    return model




def build_discriminator_semi(latent_dim):
    print "type of latent_dim", type(latent_dim)
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha =0.2))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation = 'softmax'))
    model.summary()
    encoded_repr = Input(shape = (latent_dim,))
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


if __name__ == '__main__':

    input_dim = 784
    latent_dim = 10
    output_dim = input_dim
    n_classes = 10  # 10 classes for MNIST
    discriminator_gauss= build_discriminator_gauss(latent_dim)
    discriminator_cat = build_discriminator_cat(n_classes)
    optimizer = Adam(0.0002, 0.5)
    corruption_level = 0.05


    discriminator_gauss.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    discriminator_cat.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])



    encoder, encoder_cat = build_encoder(input_dim, latent_dim, n_classes = n_classes)
    decoder = build_decoder(output_dim, latent_dim + n_classes)



    input = Input(shape = (input_dim,))
    z = encoder(input)


    y_cat = encoder_cat(input)


    encoded_repr = concatenate([y_cat, z])
    reconstructed_input = decoder(encoded_repr)



    discriminator_gauss.trainable = False
    discriminator_cat.trainable = False

    validity_gauss = discriminator_gauss(z)
    validity_cat = discriminator_cat(y_cat)



    supervised_classifier = Model(input = input, output = y_cat)
    supervised_classifier.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    adversarial_gauss = Model(input, outputs = [reconstructed_input, validity_gauss])
    adversarial_gauss.compile(loss = ['mse', 'binary_crossentropy'], loss_weights=[0.999,0.001], optimizer = optimizer)

    adversarial_cat = Model(input, outputs = [reconstructed_input, validity_cat])
    adversarial_cat.compile(loss = ['mse', 'binary_crossentropy'], loss_weights=[0.999,0.001], optimizer = optimizer)



    # training
    batch_size = 100
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    X_train = X_train.reshape(X_train.shape[0], 28*28*1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=3)
    X_test = X_test.reshape(X_test.shape[0],28*28*1)
    y_test = np.eye(10)[y_test]


    # adversarial ground truth
    valid = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))

    n_epochs = 100
    sample_interval = 200
    n_iter = n_epochs*X_train.shape[0]/batch_size

    print("number of iterations {}".format(n_iter))

    for epoch in range(n_iter):

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        y_batch = np.eye(n_classes)[y_batch]

        X_batch = X_batch.reshape(batch_size, np.prod(X_batch.shape[1:]))
        X_batch_noise = X_batch  + np.random.normal(0,1,size = X_batch.shape)*corruption_level
        latent_fake = encoder.predict(X_batch_noise)
        latent_real = np.random.normal(size = (batch_size, latent_dim))

        real_cat_dist = np.random.randint(low=0, high=10, size=batch_size)
        real_cat_dist = np.eye(n_classes)[real_cat_dist]
        fake_cat_dist = encoder_cat.predict(X_batch)

        # Train Gaussian AN
        d_loss_gauss_real = discriminator_gauss.train_on_batch(latent_real, valid)
        d_loss_gauss_fake = discriminator_gauss.train_on_batch(latent_fake, fake)
        d_loss_gauss = 0.5*np.add(d_loss_gauss_real, d_loss_gauss_fake)
        g_loss_gauss = adversarial_gauss.train_on_batch(X_batch_noise, [X_batch, valid])

        # Train Cat AN
        d_loss_cat_real = discriminator_cat.train_on_batch(real_cat_dist, valid)
        d_loss_cat_fake = discriminator_cat.train_on_batch(fake_cat_dist, valid)


        d_loss_cat_fake = discriminator_gauss.train_on_batch(latent_fake, fake)
        d_loss_cat = 0.5*np.add(d_loss_gauss_real, d_loss_gauss_fake)
        g_loss_cat = adversarial_cat.train_on_batch(X_batch_noise, [X_batch, valid])

        supervised_loss = supervised_classifier.train_on_batch(X_batch, y_batch)
        test_accuracy =  supervised_classifier.evaluate(X_test, y_test, verbose = 0)


        if epoch%sample_interval == 0:
            print ("%d [D loss Gauss: %f, acc: %.2f%%] [G loss Gauss: %f, mse: %f]" % (epoch, d_loss_gauss[0], 100*d_loss_gauss[1], g_loss_gauss[0], g_loss_gauss[1]))
            print ("%d [D loss Cat: %f, acc: %.2f%%] [G loss Cat: %f, mse: %f]" % (epoch, d_loss_cat[0], 100*d_loss_cat[1], g_loss_cat[0], g_loss_cat[1]))
            print ("%d [Supervised Loss : %f, acc: %.2f%%] " % (epoch, supervised_loss[0], 100*supervised_loss[1]))
            print ("%d [Validation Loss : %f, acc: %.2f%%] " % (epoch, test_accuracy[0], 100*test_accuracy[1]))
            print ("_____________________________________________")

        if epoch%2000 == 0:
            sample_images(epoch, latent_dim, decoder, X_batch)

        # Train semi-supervised classifier
