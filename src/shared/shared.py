""" 
Credit for the original design goes to:
Rachael Hwee Ling Sim, Yehong Zhang, Bryan Kian Hsiang Low, and Patrick Jaillet. Collaborative bayesian
optimization with fair regret. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International
Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 9691–
9701. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/sim21b.html.
Original GitHub: https://github.com/YehongZ/CollaborativeBO

"""
# Adapted to allow for modularisation and sharing across global, local and, CBO approaches
# Maintained the tf = 2.0 and old Keras API structure to allow for a cleanar comparison between the original
# CBO paper, the verification experiments conducted under CBO_Flower_Simulation, and the other two approaches.


# Make TensorFlow logs less verbose
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
import pickle
import numpy as np

from .dataset import*

def get_uncompiled_original_CBO_CNN(x_shape, l2_regular,  num_classes = 10,  dropout_rate = 0.0,  conv_filters = 16,  dense_units = 8, kernel_size = 3, pool_size = 3):
    model = Sequential()
    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                        input_shape=x_shape, kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def get_compiled_original_CBO_CNN(x_shape, learning_rate, learning_rate_decay, l2_regular):
    model = get_uncompiled_original_CBO_CNN(x_shape=x_shape, l2_regular=l2_regular)
    opt = RMSprop(lr=learning_rate, decay=learning_rate_decay)
    model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    return model


def get_epoch_opt_compiled_original_CBO_CNN(x_shape, num_classes = 10,  dropout_rate = 0.0,  conv_filters = 16,  dense_units = 8, kernel_size = 3, pool_size = 3):
    model = Sequential()
    model.add( Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                        input_shape=x_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(dense_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = RMSprop()
    model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    return model


def load_partitioned_FEMINST_data(client_cnt=10):
    data =  pickle.load(open("/home/ubuntu/r244_alex/R244_Project/src/data/emnist_data_mixed.pkl", "rb"))
    X_trains, X_tests, Y_trains, Y_tests = [], [], [], []
    for i in range(0,client_cnt):
        X_train, X_test, Y_train, Y_test = data[i][0], data[i][1], data[i][2], data[i][3]
        
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = 1 - X_train
        X_test = 1 - X_test
        X_trains.append(X_train)
        X_tests.append(X_test)
        Y_trains.append(Y_train)
        Y_tests.append(Y_test)

    X_trains, X_tests, Y_trains, Y_tests = np.array(X_trains, dtype=object), np.array(X_tests, dtype=object), np.array(Y_trains, dtype=object), np.array(Y_tests, dtype=object)
    X_eval_test = np.array(    [  item for sublist in  X_tests for item in sublist], dtype=np.float32)
    
    Y_eval_test = np.array(    [  item for sublist in  Y_tests for item in sublist], dtype=np.float32)
    return X_trains, X_tests, Y_trains, Y_tests, X_eval_test, Y_eval_test
def one_hot_encode(i, num_classes):
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[int(i)] = 1.0
    return vec

def get_iid_controlled_data_generator(iid_frac):
    def load_iid_controlled_FEMINST_data(client_cnt=10):
        nonlocal iid_frac
        data =  pickle.load(open("/home/ubuntu/r244_alex/R244_Project/src/data/emnist_data_mixed.pkl", "rb"))
        X_trains, X_tests, Y_trains, Y_tests = [], [], [], []
        for i in range(0,client_cnt):
            X_train, X_test, Y_train, Y_test = data[i][0], data[i][1], data[i][2], data[i][3]

            X_trains.append(X_train)
            X_tests.append(X_test)
            Y_trains.append(Y_train)
            Y_tests.append(Y_test)

        X_trains, X_tests, Y_trains, Y_tests = np.array(X_trains, dtype=object), np.array(X_tests, dtype=object), np.array(Y_trains, dtype=object), np.array(Y_tests, dtype=object)
        x_test = np.array([item for sublist in  X_tests for item in sublist], dtype=np.float32)[:1120]
        
        y_test = np.array(    [  np.argmax(item) for sublist in  Y_tests for item in sublist], dtype=np.float32)[:1120]

        x_train = np.array(    [  item for sublist in  X_trains for item in sublist], dtype=np.float32)[:9340]
  
        y_train = np.array(    [  np.argmax(item)  for sublist in Y_trains for item in sublist], dtype=np.float32)[:9340]

        (train_partitions, test_partitions), _ = create_partitioned_dataset(
            ((x_train, y_train), (x_test, y_test)), iid_frac, 10)
        for i in range(len(train_partitions)):
            ret_x_train, ret_y_train = train_partitions[i]
            ret_y_train = np.array([one_hot_encode(it,10) for it in ret_y_train], dtype=np.float32)
            ret_x_test, ret_y_test = test_partitions[i]
            ret_y_test = np.array([one_hot_encode(it, 10) for it in ret_y_test], dtype=np.float32)
            X_trains[i] = ret_x_train
            Y_trains[i] = ret_y_train
            X_tests[i] = ret_x_test
            Y_tests[i] =ret_y_test
    
        X_trains, X_tests, Y_trains, Y_tests = np.array(X_trains, dtype=object), np.array(X_tests, dtype=object), np.array(Y_trains, dtype=object), np.array(Y_tests, dtype=object)
        X_eval_test = np.array(    [  item for sublist in  X_tests for item in sublist], dtype=np.float32)
        
        Y_eval_test = np.array(    [  item for sublist in  Y_tests for item in sublist], dtype=np.float32)
        return X_trains, X_tests, Y_trains, Y_tests, X_eval_test, Y_eval_test

    return load_iid_controlled_FEMINST_data
