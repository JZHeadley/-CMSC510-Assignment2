#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarodz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf;


minimum=[-.25,2]

# THIS SCRIPT PERFORMS PROJECTED GRADIENT DESCENT ON FUNCTION F, 
# ASSUMING Q(the feasible region) is w1>=0, w2>=0


def f(w):
    shiftedW=w-np.array(minimum);
    return tf.reduce_sum(tf.multiply(shiftedW,shiftedW));

#define starting value of W for gradient descent
#here, W is a 2D vector
initialW=np.random.randn(2)

#create a shared variable (i.e. a variable that persists between calls to a tensorflow function)
w = tf.Variable(initialW,name="w");

#define output of applying f to w
#out goal will be to minimize f(w), i.e. find w with lowest possible f(w)
z=f(w);

# if you want more accurate result, replace step size 0.01 with something smaller
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(z)


#initialize tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

with sess:
    # hard-coded number of steps, could be too little, may need to be increased
    for i in range(300):
      #perform gradient step
      train.run();
      #get the numpy vector with current value of w
      w_value=w.eval();
      # run proximal operator (here it's simple, just replace negative values with 0)
      new_w_value=np.maximum(w_value,0);
      print((w_value,new_w_value))
      # update tensorflow value using numpy value
      new_w_assign = tf.assign(w,new_w_value);
      sess.run(new_w_assign);

#sess.close()

print("True minimum: "+str(np.maximum(minimum,0)));
print("Found minimum:"+str(new_w_value));
