
import tensorflow as tf
import numpy as np
import datetime
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras.datasets import mnist


import matplotlib
from keras.datasets import mnist
import argparse
import sys
import os
#sys.path.append(os.getcwd() + '/autoencoder_single')
sys.path.append(os.getcwd())
from classifier import*
from helper_funcs import*


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The autoencoder network
def encoder(x, input_dim, z_dim, n_l1, n_l2, n_labels,reuse=False, supervised=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :param supervised: True -> returns output without passing it through softmax,
                       False -> returns output after passing it through softmax.
    :return: tensor which is the classification output and a hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
        latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
        cat_op = dense(e_dense_2, n_l2, n_labels, 'e_label')
        if supervised:
            softmax_label = tf.nn.softmax(logits=cat_op, name='e_softmax_label')
        else:
            softmax_label = cat_op
        return softmax_label, latent_variable


def decoder(x, input_dim, z_dim, n_l1, n_l2, n_labels, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.relu(dense(x, z_dim + n_labels, n_l2, 'd_dense_1'))
        d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
        output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return output


def discriminator_gauss(x, z_dim, n_l1, n_l2, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given gaussian distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator_Gauss'):
        dc_den1 = tf.nn.relu(dense(x, z_dim, n_l1, name='dc_g_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_g_den2'))
        output = dense(dc_den2, n_l2, 1, name='dc_g_output')
        return output


def discriminator_categorical(x, n_l1, n_l2, n_labels, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given categorical distribution.
    :param x: tensor of shape [batch_size, n_labels]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator_Categorial'):
        dc_den1 = tf.nn.relu(dense(x, n_labels, n_l1, name='dc_c_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_c_den2'))
        output = dense(dc_den2, n_l2, 1, name='dc_c_output')
        return output



def load_dataset(index, rhino_root):

    all_subjects = np.array(os.listdir(rhino_root + '/scratch/tphan/joint_classifier/FR1/'))
    subject = all_subjects[index]
    print subject

    subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
    dataset = joblib.load(subject_dir)
    dataset_auto = get_session_data(subject, rhino_root)
    dataset_enc = select_phase(dataset)

    return dataset_enc, dataset_auto




def next_batch(x,batch_size, y = None):
    """
    Used to return a random batch from the given inputs.
    :param x: Input images of shape [None, 784]
    :param y: Input labels of shape [None, 10]
    :param batch_size: integer, batch size of images and labels to return
    :return: x -> [batch_size, 784], y-> [batch_size, 10]
    """
    random_index = np.random.randint(0, x.shape[0], batch_size)
    if np.sum(y) != None:
        return x[random_index], y[random_index]
    else:
        return x[random_index]


def train(X_tl, y_tl, X_tul, X_vl, y_vl, train_model=True):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """


    # Saving the model
    step = 0
    sess = tf.Session()
    sess.run(init)

    #x_l, y_l = mnist.test.next_batch(n_labeled)

    for i in range(n_epochs):
        n_batches = int(n_labeled / batch_size)
        print("------------------Epoch {}/{}------------------".format(i, n_epochs))
        for b in range(1, n_batches + 1):
            z_real_dist = np.random.randn(batch_size, z_dim) * 5.
            real_cat_dist = np.random.randint(low=0, high=2, size=batch_size)
            real_cat_dist = np.eye(n_labels)[real_cat_dist]
            #batch_x_ul, _ = mnist.train.next_batch(batch_size)

            batch_x_ul = next_batch(X_tul, batch_size)

            batch_x_l, batch_y_l = next_batch(X_tl, batch_size, y_tl)

            batch_y_l_cat = np.eye(n_labels)[batch_y_l]

            sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x_ul, x_target: batch_x_ul})
            sess.run(discriminator_g_optimizer,
                     feed_dict={x_input: batch_x_ul, x_target: batch_x_ul, real_distribution: z_real_dist})
            sess.run(discriminator_c_optimizer,
                     feed_dict={x_input: batch_x_ul, x_target: batch_x_ul,
                                categorial_distribution: real_cat_dist})
            sess.run(generator_optimizer, feed_dict={x_input: batch_x_ul, x_target: batch_x_ul})
            sess.run(supervised_encoder_optimizer, feed_dict={x_input_l: batch_x_l, y_input: batch_y_l_cat})


            # if b % 20 == 0:
            #
            #     a_loss, d_g_loss, d_c_loss, g_loss, s_loss = sess.run(
            #         [autoencoder_loss, dc_g_loss, dc_c_loss, generator_loss, supervised_encoder_loss],
            #         feed_dict={x_input: batch_x_ul, x_target: batch_x_ul,
            #                    real_distribution: z_real_dist, y_input: batch_y_l_cat, x_input_l: batch_x_l,
            #                    categorial_distribution: real_cat_dist})
            #
            #
            #     #writer.add_summary(summary, global_step=step)
            #     print("Epoch: {}, iteration: {}".format(i, b))
            #     print("Autoencoder Loss: {}".format(a_loss))
            #     print("Discriminator Gauss Loss: {}".format(d_g_loss))
            #     print("Discriminator Categorical Loss: {}".format(d_c_loss))
            #     print("Generator Loss: {}".format(g_loss))
            #     print("Supervised Loss: {}\n".format(s_loss))
            step += 1
        acc = 0
        num_batches = int(X_tl.shape[0]/batch_size)

    y_tl_cat = np.eye(n_labels)[y_tl]
    y_vl_cat = np.eye(n_labels)[y_vl]
    encoder_train_pred = sess.run(encoder_output_label_, feed_dict = {x_input_l:X_tl, y_input:y_tl_cat})
    encoder_val_pred = sess.run(encoder_output_label_, feed_dict = {x_input_l:X_vl, y_input:y_vl_cat})
    return encoder_val_pred[:,1]

        # for j in range(num_batches):
        #     # Classify unseen validation data instead of test data or train data
        #     batch_x_l, batch_y_l = next_batch(X_vl, batch_size, y_vl)
        #     batch_y_l_cat = np.eye(n_labels)[batch_y_l]
        #     encoder_acc = sess.run(accuracy, feed_dict={x_input_l: batch_x_l, y_input: batch_y_l_cat})
        #     acc += encoder_acc
        # acc /= num_batches



if __name__ == '__main__':


    args = sys.argv
    print args
    index = int(args[1])
    penalty = pow(10,-int(args[2]))

    n_labels = 2

    rhino_root = '/Volumes/RHINO'
    #index =0
    dataset_enc, dataset_auto = load_dataset(index, rhino_root)
    sessions = np.unique(dataset_enc['session'])

    dataset_enc_temp = dataset_enc
    dataset_enc_temp['X'] = normalize_sessions(dataset_enc_temp['X'], dataset_enc_temp['session'])
    result_current = run_loso_xval(dataset_enc_temp, classifier_name = 'current', search_method = 'tpe', type_of_data = 'rand',  feature_select= 0,  adjusted = 1, C_factor = 1.0)

    print result_current

    dataset_enc['X'] = scale_sessions(dataset_enc['X'], dataset_enc['session'], dataset_enc['X'], dataset_enc['session'])
    dataset_auto['X'] = scale_sessions(dataset_auto['X'], dataset_auto['session'], dataset_auto['X'], dataset_auto['session'])



    # training
    batch_size = 24

    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    sessions = np.unique(dataset_enc['session'])

    probs_all = []
    y_all = []

        # Parameters
    input_dim = dataset_enc['X'].shape[1]
    n_l1 = 64
    n_l2 = 64
    z_dim = 10
    batch_size = 12
    n_epochs = 1000
    learning_rate = 0.001
    beta1 = 0.9
    n_labels = 2
    n_labeled = dataset_enc['X'].shape[0]



    # Placeholders for input data and the targets
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input')
    x_input_l = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Labeled_Input')
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, n_labels], name='Labels')
    x_target = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Target')
    real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='Real_distribution')
    categorial_distribution = tf.placeholder(dtype=tf.float32, shape=[None, n_labels],
                                             name='Categorical_distribution')
    manual_decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim + n_labels], name='Decoder_input')

    # Reconstruction Phase
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output_label, encoder_output_latent = encoder(x_input, input_dim, z_dim, n_l1, n_l2, n_labels)
        # Concat class label and the encoder output
        decoder_input = tf.concat([encoder_output_label, encoder_output_latent], 1)
        decoder_output = decoder(decoder_input, input_dim, z_dim, n_l1, n_l2, n_labels)



    # Regularization Phase
    with tf.variable_scope(tf.get_variable_scope()):
        d_g_real = discriminator_gauss(real_distribution, z_dim, n_l1, n_l2)
        d_g_fake = discriminator_gauss(encoder_output_latent, z_dim, n_l1, n_l2,  reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        d_c_real = discriminator_categorical(categorial_distribution, n_l1, n_l2, n_labels)
        d_c_fake = discriminator_categorical(encoder_output_label, n_l1, n_l2, n_labels, reuse=True)

    # Semi-Supervised Classification Phase
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output_label_, _ = encoder(x_input_l, input_dim, z_dim, n_l1, n_l2, n_labels, reuse=True, supervised=True)

    # Generate output images
    # with tf.variable_scope(tf.get_variable_scope()):
    #     decoder_image = decoder(manual_decoder_input, reuse=True)

    # Classification accuracy of encoder

    pred = tf.nn.softmax(encoder_output_label_)

    correct_pred = tf.equal(tf.argmax(encoder_output_label_, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Gaussian Discriminator Loss
    dc_g_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_g_real), logits=d_g_real))
    dc_g_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_g_fake), logits=d_g_fake))
    dc_g_loss = dc_g_loss_fake + dc_g_loss_real

    # Categorical Discrimminator Loss
    dc_c_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_c_real), logits=d_c_real))
    dc_c_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_c_fake), logits=d_c_fake))
    dc_c_loss = dc_c_loss_fake + dc_c_loss_real

    # Generator loss
    generator_g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_g_fake), logits=d_g_fake))
    generator_c_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_c_fake), logits=d_c_fake))
    generator_loss = generator_c_loss + generator_g_loss

    # Supervised Encoder Loss
    supervised_encoder_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=encoder_output_label_))

    all_variables = tf.trainable_variables()
    dc_g_var = [var for var in all_variables if 'dc_g_' in var.name]
    dc_c_var = [var for var in all_variables if 'dc_c_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                       beta1=beta1).minimize(dc_g_loss, var_list=dc_g_var)
    discriminator_c_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                       beta1=beta1).minimize(dc_c_loss, var_list=dc_c_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=en_var)
    supervised_encoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                          beta1=beta1).minimize(supervised_encoder_loss,
                                                                                var_list=en_var)

    init = tf.global_variables_initializer()



    for session in sessions:

        all_sess_mask = dataset_auto['session'] == session
        enc_sess_mask = dataset_enc['session'] == session
        X_tul = dataset_auto['X'][~all_sess_mask]
        print "number of samples = ", X_tul.shape[0]
        X_tl= dataset_enc['X'][~enc_sess_mask]
        y_tl = dataset_enc['y'][~enc_sess_mask]
        X_vl = dataset_enc['X'][enc_sess_mask]
        y_vl = dataset_enc['y'][enc_sess_mask]
        y_test_labeled_cat = np.eye(n_labels)[y_vl]



        # Saving the model
        step = 0
        sess = tf.Session()
        sess.run(init)

        #x_l, y_l = mnist.test.next_batch(n_labeled)

        for i in range(n_epochs):
            n_batches = int(n_labeled / batch_size)
            print("------------------Epoch {}/{}------------------".format(i, n_epochs))
            for b in range(1, n_batches + 1):
                z_real_dist = np.random.randn(batch_size, z_dim) * 5.
                real_cat_dist = np.random.randint(low=0, high=2, size=batch_size)
                real_cat_dist = np.eye(n_labels)[real_cat_dist]
                #batch_x_ul, _ = mnist.train.next_batch(batch_size)

                batch_x_ul = next_batch(X_tul, batch_size)

                batch_x_l, batch_y_l = next_batch(X_tl, batch_size, y_tl)

                batch_y_l_cat = np.eye(n_labels)[batch_y_l]

                sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x_ul, x_target: batch_x_ul})
                sess.run(discriminator_g_optimizer,
                         feed_dict={x_input: batch_x_ul, x_target: batch_x_ul, real_distribution: z_real_dist})
                sess.run(discriminator_c_optimizer,
                         feed_dict={x_input: batch_x_ul, x_target: batch_x_ul,
                                    categorial_distribution: real_cat_dist})
                sess.run(generator_optimizer, feed_dict={x_input: batch_x_ul, x_target: batch_x_ul})
                sess.run(supervised_encoder_optimizer, feed_dict={x_input_l: batch_x_l, y_input: batch_y_l_cat})


                if b % 20 == 0:

                    a_loss, d_g_loss, d_c_loss, g_loss, s_loss = sess.run(
                        [autoencoder_loss, dc_g_loss, dc_c_loss, generator_loss, supervised_encoder_loss],
                        feed_dict={x_input: batch_x_ul, x_target: batch_x_ul,
                                   real_distribution: z_real_dist, y_input: batch_y_l_cat, x_input_l: batch_x_l,
                                   categorial_distribution: real_cat_dist})


                    #writer.add_summary(summary, global_step=step)
                    # print("Epoch: {}, iteration: {}".format(i, b))
                    # print("Autoencoder Loss: {}".format(a_loss))
                    # print("Discriminator Gauss Loss: {}".format(d_g_loss))
                    # print("Discriminator Categorical Loss: {}".format(d_c_loss))
                    # print("Generator Loss: {}".format(g_loss))
                    # print("Supervised Loss: {}\n".format(s_loss))
                step += 1
            acc = 0
            num_batches = int(X_tl.shape[0]/batch_size)

        y_tl_cat = np.eye(n_labels)[y_tl]
        y_vl_cat = np.eye(n_labels)[y_vl]
        encoder_train_pred = sess.run(encoder_output_label_, feed_dict = {x_input_l:X_tl, y_input:y_tl_cat})
        encoder_val_pred = sess.run(encoder_output_label_, feed_dict = {x_input_l:X_vl, y_input:y_vl_cat})

        probs_all.append(encoder_val_pred[:,1])
        y_all.append(y_vl)








    label_all = np.concatenate(y_all)
    probs_all = np.concatenate(probs_all)
    auc_auto = sklearn.metrics.roc_auc_score(label_all,probs_all)
    print "auto auc", auc_auto
    print "current auc", result_current
