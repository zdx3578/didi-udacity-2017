SEED = 202

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
sess = tf.Session()


import keras
from keras import backend as K
K.set_session(sess)
assert(K._BACKEND=='tensorflow')

#K.learning_phase() #0:test,  1:train

from keras.models import Sequential, Model
from keras.layers import Deconvolution2D, Convolution2D, Cropping2D, Cropping1D, Input, merge
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,SReLU,ELU
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import initializations

from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler




# num libs
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import math




# my libs
from utility.file import *
