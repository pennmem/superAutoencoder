import tensorflow as tf
import numpy as np


def sigmoid(X):
    return 1.0/(1.0 + np.exp(-X))


def softmax(X, copy=True):
    """
    Calculate the softmax function.

    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)

    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.

    Parameters
    ----------
    X : array-like, shape (M, N)
        Argument to the logistic function

    copy : bool, optional
        Copy X or not.

    Returns
    -------
    out : array, shape (M, N)
        Softmax function evaluated at every point in x
    """
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X



class noisy_LogisticRegression(object):

    def __init__(self, **kwargs):
        #print kwargs
        self.C = kwargs['C']
        self.learning_rate = kwargs['learning_rate']
        self.noise_penalty = kwargs['noise_penalty']
        self.sigma_noise = kwargs['sigma_noise']
        self.batch_size = kwargs['batch_size']
        self.momentum = kwargs['momentum']
        self.max_iter = kwargs['max_iter']


    def get_model_ouput(self):
        model_output = tf.add(tf.matmul(self.x_data,self.W), self.b)
        deviance = tf.reduce_sum(tf.square(tf.matmul(self.x_data-self.x_tilde_data,self.W)))
        l2regularization = tf.reduce_sum(tf.square(self.W)) + tf.square(self.b) # penalize intercept as well
        llhood = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(logits = model_output, targets = self.y_target, pos_weight= self.pos_weight))

        return (llhood, deviance, l2regularization)

    def set_loss(self):
        llhood, deviance, l2regularization = self.get_model_ouput()
        self.loss = self.C*llhood + 0.5*l2regularization + self.noise_penalty*deviance


    def get_params(self, deep = True): # implement get params
        return dict(zip(['C', 'learning_rate', 'noise_penalty',
                         'sigma_noise', 'batch_size', 'momentum', 'max_iter'],[self.C,self.learning_rate, self.noise_penalty, self.sigma_noise, self.batch_size,
                         self.momentum, self.max_iter]))


    def predict_proba(self,X):  # predict probabilities

        if not hasattr(self, "coef_"):
            raise ValueError("Call fit before prediction")

        predict_linear = np.matmul(X,self.coef_) + self.intercept_
        predict_prob = sigmoid(predict_linear)

        return np.concatenate([1-predict_prob,predict_prob], axis = 1)



    def fit(self,X,y):

        assert X.shape[0] == len(y), "Dimensions mismatched"
        self.X = X
        self.y = y
        self.p = X.shape[1]
        self.N = X.shape[0]
        self.x_data = tf.placeholder(shape = [None, self.p ], dtype = tf.float32)
        self.x_tilde_data = tf.placeholder(shape = [None, self.p ], dtype = tf.float32)
        self.y_target = tf.placeholder(shape = [None,1], dtype = tf.float32)
        self.my_opt = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = self.momentum, use_nesterov= True)
        self.loss_vec = []
        # variables
        self.W = tf.Variable(tf.random_normal(shape =[self.p,1]))  # slope
        self.b = tf.Variable(tf.random_normal(shape = [1,1]))  # intercept

        n_pos = np.sum(y)*1.0
        n_neg = len(y) - n_pos
        self.pos_weight = n_neg/n_pos

        self.set_loss()
        #print self.loss
        train_step = self.my_opt.minimize(self.loss)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        #init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(self.max_iter):
            rand_index = np.random.choice(len(self.X), size = self.batch_size)
            rand_x = self.X[rand_index]
            rand_x_tilde = rand_x + np.random.normal(0,self.sigma_noise, rand_x.shape) # add noisy component to training
            rand_y = np.transpose([self.y[rand_index]])
            sess.run(train_step, feed_dict = {self.x_data:rand_x, self.x_tilde_data:rand_x_tilde,self.y_target:rand_y})
            temp_loss = sess.run(self.loss, feed_dict={self.x_data:rand_x, self.x_tilde_data:rand_x,  self.y_target:rand_y})
            self.loss_vec.append(temp_loss)
            #print temp_loss
            #if (i+1)%300==0:
                #print('Loss = ' + str(temp_loss))

        self.coef_ = sess.run(self.W)
        self.intercept_ = sess.run(self.b)

        # return the classifier
        return self

