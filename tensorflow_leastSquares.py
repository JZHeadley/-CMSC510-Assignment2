#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarodz
"""
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from support import *
# for reproducibility between runs
np.random.seed(123)


def makeFakeDataset():
    sCnt = 10000
    sCnt2 = 2000

    numberOfFeatures = 784

    # true parameters w and b
    true_w = np.zeros((numberOfFeatures, 1))
    true_w[0] = -0.5
    true_w[1] = 1.3
    true_w[2] = 1.3
    true_b = -0.3

    # sample some random point in feature space
    X = np.random.randn(sCnt+sCnt2, numberOfFeatures).astype(dtype='float32')

    # calculate u=w^Tx+b
    true_u = np.dot(X, true_w) + true_b

    # add gaussian noise
    Y = true_u + 0.01*np.random.randn(sCnt+sCnt2, 1)

    # split into training and test set
    return X[0:sCnt, :], Y[0:sCnt, :], X[sCnt:sCnt+sCnt2, :], Y[sCnt:sCnt+sCnt2, :]


# we use the dataset with x_train being the matrix "n by fCnt"
# with samples as rows, and the  features as columns
# y_train is the true value of dependent variable, we have it as a matrix "n by 1"
# V00746112
classValue1 = 1
classValue2 = 2
percTrain = 1
percTest = 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.set_printoptions(linewidth=250)

x_train, y_train = extractMine(x_train, y_train,  classValue1, classValue2)
x_test, y_test = extractMine(x_test, y_test, classValue1, classValue2)

# numToTrainOn = int(percTrain*x_train.__len__())
# numToTestOn = int(percTest*x_test.__len__())

# x_train = np.array(x_train[:numToTrainOn])
# y_train = np.array(y_train[:numToTrainOn])
# x_test = np.array(x_test[:numToTestOn])
# y_test = np.array(y_test[:numToTestOn])

# x_test_class1 = extractClass(x_test, y_test, 1)
# x_test_class2 = extractClass(x_test, y_test, 0)

x_train = flat_norm(x_train)
x_test = flat_norm(x_test)

x_train, y_train, x_test, y_test = makeFakeDataset()
n_train = x_train.shape[0]
fCnt = x_train.shape[1]


# START OF LEARNING

# number of epchos. 1 epoch is when all training data is seen
n_epochs = 100

# number of samples to present as micro-batch
# could be just n_train
# or if dataset is large, could be a small chunk of if
batch_size = 128
# ^^^ says: present the training set in chunks (micro-batches) of 128 samples


# define variables for tensorflow

# define and initialize shared variables
# (the variable persist, they encode the state of the classifier throughout learning via gradient descent)
# w is the feature weights, a [fCnt x 1] vector
initialW = np.random.rand(fCnt, 1).astype(dtype='float32')
w = tf.Variable(initialW, name="w")
tf.summary.histogram('w', w)

# b is the bias, so just a single number
initialB = 0.0
b = tf.Variable(initialB, name="b")
tf.summary.scalar('b', b)

# define non-shared/placeholder variable types
# x will be our [#samples x #features] matrix of all training samples
x = tf.placeholder(dtype=tf.float32, name='x')
# y will be our vector of dependent variable for all training samples
y = tf.placeholder(dtype=tf.float32, name='y')


# set up new variables that are functions/transformations of the above
# predicted class for each sample (a vector)
# tf.matmul(x,w) is a vector with #samples entries
# even though b is just a number, + will work (through "broadcasting")
# b will be "replicated" #samples times to make both sides of + have same dimension
# thre result is a vector with #samples entries
predictions = tf.matmul(x, w)+b
tf.summary.histogram('predictions', predictions)

# loss (square error of prediction) for each sample (a vector)
loss = tf.square(y-predictions)
# loss = tf.losses.log_loss(labels=y, predictions=predictions)
# tf.summary.scalar('loss', loss)
# risk over all samples (a number)
risk = tf.reduce_mean(loss)
tf.summary.scalar('risk', risk)
# define which optimizer to use
optimizer = tf.train.GradientDescentOptimizer(0.01)
# tf.summary.histogram('optimizer', optimizer)
train = optimizer.minimize(risk)
# tf.summary.histogram('train', train)
# create a tensorflow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()

summaries_dir = "/tmp/tensorflow/mnist"
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
tf.global_variables_initializer().run(session=sess)

# calculate and print Mean Squared Error on the full test set, using initial (random) w,b
y_pred = sess.run([predictions], feed_dict={x: x_test, y: y_test})[0]
MSE = np.mean(np.square(y_pred-y_test), axis=0)[0]
tf.summary.scalar('MSE', MSE)
print(MSE)
merged = tf.summary.merge_all()

# start the iterations of training
# 1 epoch == all data samples were presented
for i in range(0, n_epochs):
    # if dataset is large, we want to present it in chunks (called micro-batches)
    for j in range(0, n_train, batch_size):
        jS = j
        jE = min(n_train, j+batch_size)
        x_batch = x_train[jS:jE, :]
        y_batch = y_train[jS:jE, :]
        # do a step of gradient descent on the micro-batch
        _, curr_batch_risk, predBatchY = sess.run(
            [train, risk, predictions], feed_dict={x: x_batch, y: y_batch})
    # training done in this epoch
    # but, just so that the user can monitor progress, try current w,b on full test set
    summary, y_pred, curr_w, curr_b = sess.run(
        [merged, predictions, w, b], feed_dict={x: x_test, y: y_test})

    # calculate and print Mean Squared Error
    MSE = np.mean(np.mean(np.square(y_pred-y_test), axis=1), axis=0)
    train_writer.add_summary(summary, i)

    print(MSE)
print(np.transpose(curr_w))
print(curr_b)

# test_writer = tf.summary.FileWriter(summaries_dir + '/test')