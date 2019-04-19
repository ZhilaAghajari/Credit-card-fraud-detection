import tensorflow as tf 
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

seed = 1987 # my birth year
tf.set_random_seed(seed)
np.random.seed(seed)
checkpoint_file='./nn_model.ckpt'

class NeuralNetowrk(BaseEstimator, ClassifierMixin):
    def fully_connected_layer(self, input, out_dim, act_fn=tf.nn.relu):
        out = tf.contrib.layers.fully_connected(
            inputs=input, num_outputs=out_dim, activation_fn=act_fn, 
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.initializers.zeros()
        )
        return out

    def build_network(self):
        outputs = [self.X]
        for layer_size in self.hidden_layer_sizes:
            out = self.fully_connected_layer(outputs[-1], layer_size, act_fn=tf.nn.elu)
            outputs.append(out)
        self.logits = self.fully_connected_layer(outputs[-1], 2, act_fn=None) # 2 as our task is binary classification

        self.y_pred = tf.nn.softmax(self.logits)
        self.y_cls = tf.argmax(self.y_pred, axis=1)
        self.y_true = tf.argmax(self.y, axis=1)

        self.equals = tf.cast(tf.equal(tf.cast(self.y_cls, dtype=tf.float32), tf.cast(self.y_true, dtype=tf.float32)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(self.equals)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits))

        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def shuffle(self, X, y):
        data = np.hstack([X, y])
        shuffled = np.random.permutation(data)
        X, y = shuffled[:, :-2], shuffled[:,-2:]
        return X, y

    def train_neural_network(self, epochs):
        self.build_network()
        init = tf.global_variables_initializer()

        in_X = self.X_train
        if isinstance(in_X, pd.DataFrame):
            in_X = in_X.values
        in_y = self.y_train
        if len(in_y.shape) == 1 or in_y.shape[1] == 1:
            in_y = one_hot_encoding(in_y)

        in_X, in_y = self.shuffle(in_X, in_y) 
        var_len = int(len(in_X) * 0.05)
        var_X, var_y, in_X, in_y = in_X[:var_len], in_y[:var_len], in_X[var_len:], in_y[var_len:]
              
        self.model_saver = tf.train.Saver()

        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)
        max_val_acc = 0
        min_val_loss = 1e9

        for epoch in range(epochs):
            in_X, in_y = self.shuffle(in_X, in_y)
            batch_size = 50
            start_batch = 0
            epoch_loss = 0
            while start_batch < len(in_X):
                batch_X, batch_y = in_X[start_batch:start_batch + batch_size], in_y[start_batch:start_batch + batch_size]
                _, batch_loss = self.sess.run([self.opt, self.loss], feed_dict={self.X: batch_X, self.y: batch_y})
                epoch_loss += batch_loss
                start_batch += batch_size

            val_loss, val_acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.X:var_X, self.y:var_y})
            if epoch == 0 or (max_val_acc < val_acc and val_loss < min_val_loss):
                self.model_saver.save(self.sess, checkpoint_file)
            
            min_val_loss = min(val_loss, min_val_loss)
            max_val_acc = max(val_acc, max_val_acc)

    def fit(self, X_train, y_train, epochs=100, hidden_layer_sizes=[32,32,32,32]):
        self.X_train = X_train 
        self.y_train = y_train
        self.hidden_layer_sizes = hidden_layer_sizes
        self.X = tf.placeholder(tf.float32, [None, self.X_train.shape[1]], name='input')
        self.y = tf.placeholder(tf.float32, [None, 2], name='input')
        self.train_neural_network(epochs=epochs)

    def predict(self, X_test):
        self.model_saver.restore(self.sess, checkpoint_file)
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        y_cls = self.sess.run(self.y_cls, feed_dict={self.X: X_test})
        return y_cls

    def predict_proba(self, X_test):
        self.model_saver.restore(self.sess, checkpoint_file)
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        y_pred = self.sess.run(self.y_pred, feed_dict={self.X: X_test})
        return y_pred 

nn_clf = NeuralNetowrk()
knn_clf = KNeighborsClassifier(n_neighbors=4)
nb_clf = GaussianNB(var_smoothing=1e-03)
lr_clf = LogisticRegression(solver='liblinear', tol=1e-2)
dt_clf = DecisionTreeClassifier(max_depth=3)

models = [nn_clf, knn_clf, nb_clf, lr_clf, dt_clf]

if __name__ == '__main__':
    for model in models:
        tf.set_random_seed(seed)
        np.random.seed(seed)
        model.fit(X_train, y_train)
        y_cls = model.predict(X_test)
        print('Model name: {}, Accuracy: {:.2%}'.format(model.__class__.__name__, accuracy_score(y_test, y_cls)))

    tf.set_random_seed(seed)
    np.random.seed(seed)
    estimators = [(model.__class__.__name__, model) for model in models]
    vt_clf = VotingClassifier(estimators=estimators, weights=[3,1,1,1,2], voting='hard')
    vt_clf.fit(X_train, y_train)
    y_cls = vt_clf.predict(X_test)
    print('Model name: {}, Accuracy: {:.2%}'.format(vt_clf.__class__.__name__, accuracy_score(y_test, y_cls)))
