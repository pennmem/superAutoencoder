# adversarial autoencoder to combine datasets across subjects


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
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

def build_encoder(input_dim, latent_dim):
    input = Input(shape = (input_dim,))
    h = Dense(512)(input)
    h = LeakyReLU(alpha=0.2)(h)
    mu = Dense(latent_dim)(h)
    log_var = Dense(latent_dim)(h)
    latent_repr = merge([mu, log_var], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),output_shape=lambda p: p[0])
    model = Model(input, latent_repr)
    model.summary()

    return model


# fix this later
def build_decoder(output_dim, latent_dim):
    model = Sequential()
    model.add(Dense(512, input_dim = latent_dim))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dense(output_dim, activation = 'tanh'))  # normalize data between -1 and 1
    model.summary()
    z = Input(shape = (latent_dim,))
    output = model(z)

    return Model(z, output)


def discriminator(latent_dim):

    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha =0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    encoded_repr = Input(shape = (latent_dim,))
    validity = model(encoded_repr)

    model = Model(encoded_repr, validity)
    model.summary()

    return model


input_dim = 1000
latent_dim = 256
output_dim = input_dim
discriminator= discriminator(latent_dim)
optimizier = Adam(0.0002, 0.5)
discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizier, metrics = ['accuracy'])
encoder = build_encoder(input_dim, latent_dim)
decoder = build_decoder(output_dim, latent_dim)

input = Input(shape = (input_dim,))
encoded_repr = encoder(input)
reconstructed_input = decoder(encoded_repr)


discriminator.trainable = False
validity = discriminator(encoded_repr)
adversarial_autoencoder = Model(input, [reconstructed_input, validity])


adversarial_autoencoder.compile(loss = ['mse', 'binary_crossentropy'], loss_weights=[0.999,0.001], optimizier = optimizier)






