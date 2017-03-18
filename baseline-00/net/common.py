SEED = 202


TOP_Y_MIN=-20  #40
TOP_Y_MAX=+20
TOP_X_MIN=0
TOP_X_MAX=40  #70.4
TOP_Z_MIN=-2.0    ###<todo> determine the correct values!
TOP_Z_MAX=0.4

TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1  #0.1
TOP_Z_DIVISION=0.4

#----------------------------------

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')


#----------------------------------

# std libs
import os
import pickle
from timeit import default_timer as timer
from datetime import datetime
import csv
import pandas as pd
import pickle

# deep learning libs
import tensorflow as tf
tf.set_random_seed(SEED)

# import keras
# from keras import backend as K
#sess = tf.Session()
# K.set_session(sess)
# assert(K._BACKEND=='tensorflow')
#K.learning_phase() #0:test,  1:train
# from keras.models import Sequential, Model
# from keras.layers import Deconvolution2D, Convolution2D, Cropping2D, Cropping1D, Input, merge
# from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation, Reshape
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import PReLU,SReLU,ELU
# from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# from keras import initializations
#
# from keras.models import load_model
# from keras.optimizers import Adam, SGD
# from keras.regularizers import l2
# from keras.callbacks import LearningRateScheduler




# num libs
import math
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)


import cv2
import matplotlib.pyplot as plt
import mayavi.mlab as mlab


# my libs
