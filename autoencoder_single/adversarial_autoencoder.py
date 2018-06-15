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


def build_discriminator(latent_dim):

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


input_dim = 784
latent_dim = 256
output_dim = input_dim
discriminator= build_discriminator(latent_dim)
optimizer = Adam(0.0002, 0.5)
discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
encoder = build_encoder(input_dim, latent_dim)
decoder = build_decoder(output_dim, latent_dim)

input = Input(shape = (input_dim,))
encoded_repr = encoder(input)
reconstructed_input = decoder(encoded_repr)

discriminator.trainable = False
validity = discriminator(encoded_repr)
adversarial_autoencoder = Model(input, outputs = [reconstructed_input, validity])
adversarial_autoencoder.compile(loss = ['mse', 'binary_crossentropy'], loss_weights=[0.999,0.001], optimizer = optimizer)


# training
batch_size = 12
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

# adversarial ground truth
valid = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

epochs = 200000

for epoch in range(epochs):

    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    imgs = imgs.reshape(batch_size, np.prod(imgs.shape[1:]))
    latent_fake = encoder.predict(imgs)
    latent_real = np.random.normal(size = (batch_size, latent_dim))

    d_loss_real = discriminator.train_on_batch(latent_real, valid)
    d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

    g_loss = adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

    if epoch%10 == 0:
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))



latent_real = np.random.normal(size = (batch_size, latent_dim))
fake_imgs = decoder.predict(latent_real)
fake_imgs = fake_imgs.reshape((12,28,28))


import matplotlib.pyplot as plt
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_train[i,:,:,0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()