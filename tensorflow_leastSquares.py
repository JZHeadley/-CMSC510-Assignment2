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
import time


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


# for reproducibility between runs
np.random.seed(123)

start_time = time.time()

# we use the dataset with x_train being the matrix "n by fCnt"
# with samples as rows, and the  features as columns
# y_train is the true value of dependent variable, we have it as a matrix "n by 1"
# V00746112
classValue1 = 1
classValue2 = 2
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.set_printoptions(linewidth=250)

x_train, y_train = extractMine(x_train, y_train,  classValue1, classValue2)
x_test, y_test = extractMine(x_test, y_test, classValue1, classValue2)
# y_train = y_train.reshape(y_train.__len__(), 1)
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
# print(x_train.shape, y_train.shape)

# x_train, y_train, x_test, y_test = makeFakeDataset()
# print(x_train.shape, y_train.shape)
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
w = tf.Variable(np.random.rand(fCnt, 1).astype(dtype='float64'), name="w")
tf.summary.histogram('w', w)

# b is the bias, so just a single number
initialB = 0.0
b = tf.Variable(initialB, name="b", dtype=tf.float64, )
tf.summary.scalar('b', b)

# define non-shared/placeholder variable types
# x will be our [#samples x #features] matrix of all training samples
x = tf.placeholder(dtype=tf.float64, name='x')
# y will be our vector of dependent variable for all training samples
y = tf.placeholder(dtype=tf.float64, name='y')


# set up new variables that are functions/transformations of the above
# predicted class for each sample (a vector)
# tf.matmul(x,w) is a vector with #samples entries
# even though b is just a number, + will work (through "broadcasting")
# b will be "replicated" #samples times to make both sides of + have same dimension
# thre result is a vector with #samples entries
predictions = tf.matmul(x, w) + b
# tf.summary.histogram('predictions', predictions)

# loss (square error of prediction) for each sample (a vector)
# loss = tf.square(y-predictions)
loss = tf.log(
    1 + tf.exp(
        tf.multiply(tf.negative(y), predictions)
    )
)
# w*x then * y look at dimensionality

# loss = tf.log(1 + tf.exp(tf.matmul(tf.matmul(tf.negative(y), tf.transpose(w)), x)))
# loss = tf.log(1 + tf.exp((-1.0 * y) * tf.transpose(w) * x))
# loss = tf.losses.log_loss(labels=y, predictions=predictions)
tf.summary.scalar('loss', loss[0][0])

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
        jE = min(n_train, j+batch_size)
        x_batch = x_train[j:jE, :]
        y_batch = y_train[j:jE, :]
        # do a step of gradient descent on the micro-batch
        _, curr_batch_risk, predBatchY, curr_w = sess.run(
            [train, risk, predictions, w], feed_dict={x: x_batch, y: y_batch})
    # training done in this epoch
        # new_w = tf.maximum(curr_w, 0)
        # new_w_assign = tf.assign(w, new_w)
        # sess.run(new_w_assign)
    print("risk:", curr_batch_risk)
    # but, just so that the user can monitor progress, try current w,b on full test set
    summary, y_pred, curr_w, curr_b = sess.run(
        [merged, predictions, w, b], feed_dict={x: x_test, y: y_test})
  # calculate and print Mean Squared Error
    # print("w is", curr_w)
    # print("b is", curr_b)
    MSE = np.mean(np.mean(np.square(y_pred-y_test), axis=1), axis=0)
    train_writer.add_summary(summary, i)
    print(MSE)

print(np.transpose(curr_w))
print(curr_b)
y_test_class1 = extractClass(x_test, y_test, 1)
y_test_class0 = extractClass(x_test, y_test, -1)

test_class = []
numClass0 = 0
numClass1 = 0

shaped_w = curr_w.reshape(curr_w.__len__(), 1)
# if positive 1 class if negative the other
u_val = np.matmul(x_test, shaped_w) + curr_b
# test_class_val = 1.0 / (1.0 + np.exp(-1.0*u_val))  # dont need this
for i in range(0, x_test.__len__()):
    if u_val[i][0] < 0:
        numClass0 += 1
        test_class.append(-1)
    elif u_val[i][0] > 0:
        numClass1 += 1
        test_class.append(1)

test_class = np.array(test_class)
y_test = y_test.reshape(1, y_test.__len__())[0]
print(test_class)
print(y_test)
print("\n")
print("That took", "{:.2f}".format(
    round(time.time()-start_time, 2)), "seconds to run")
print("We predicted we have", numClass0, "images of", classValue1, "'s.  We actually have",
      y_test_class0.__len__(), "images of", classValue1, "'s")
print("We predicted we have", numClass1, "images of", classValue2, "'s.  We actually have",
      y_test_class1.__len__(), "images of", classValue2, "'s")


print("Accuracy is", "{0:.2f}".format(round(computeAccuracy(y_test, test_class), 4)*100), "% using",
      y_train.__len__(), "training samples and", y_test.__len__(), "testing samples, each with", fCnt, "features.")
metrics = precision_recall_fscore_support(y_test, test_class, labels=[-1, 1])

print("\t|  Precision\t|  Recall\t|  FScore")
print("--------+---------------+---------------+-----------")
print("Class 1\t| ", "{0:.4f}".format(round(metrics[0][1], 4)), "\t| ", "{0:.4f}".format(
    round(metrics[1][1], 4)), "\t| ", "{0:.4f}".format(round(metrics[2][1], 4)))
print("Class 2\t| ", "{0:.4f}".format(round(metrics[0][0], 4)), "\t| ", "{0:.4f}".format(
    round(metrics[1][0], 4)), "\t| ", "{0:.4f}".format(round(metrics[2][0], 4)))
print()


# test_writer = tf.summary.FileWriter(summaries_dir + '/test')
