import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from suplement import reshape_labels, predict
from model import model

#Load data method
def load_data():
    data_tuples = []        #Return this value with 3 two-tuples
    #Load in MNIST dataset
    directory_train = './dataset/train.csv'
    directory_test = './dataset/test.csv'
    data_frame_train = pd.read_csv(directory_train)
    data_frame_test = pd.read_csv(directory_test)

    #42,000 entries, 784 input units (28*28 pixels)
    df0_label = data_frame_train['label'].to_numpy()
    df0_train = data_frame_train.drop('label', axis=1).to_numpy()

    #Training set
    y_train = np.array(df0_label[0:25200], dtype='int32')          #shape(25200,)
    X_train = np.array(df0_train[0:25200, :], dtype='float32')     #shape(25200, 784)
    data_tuples.append(shuffle(X_train, y_train))

    #Cross-validation set
    y_cval = np.array(df0_label[25200:33600], dtype='int32')     #shape(8400,)
    X_cval = np.array(df0_train[25200:33600, :], dtype='float32')  #shape(8400, 784)
    data_tuples.append(shuffle(X_cval, y_cval))

    #Testing set shape(8400, 784)
    y_test = np.array(df0_label[33600:42000], dtype='int32')     #shape(8400,)
    X_test = np.array(df0_train[33600:42000, :], dtype='float32')  #shape(8400, 784)
    data_tuples.append(shuffle(X_test, y_test))

    return data_tuples

def main():
    # Loading data
    (X_train, y_train), (X_cval, y_cval), (X_test, y_test) = load_data()

    #Scaling data
    X_train /= 255.0
    X_cval /= 255.0
    X_test /= 255.0

    m = y_train.size

    # Reshaping training/testing/cval sets

    X_train = X_train.T                                 #(784, 25200)
    y_train = reshape_labels(y_train)
    y_train = y_train.T.reshape(10, X_train.shape[1])    #(10, 25200)

    X_test = X_test.T
    #y_test = reshape_labels(y_test)
    y_test = y_test.T.reshape(1, X_test.shape[1])

    X_cval = X_cval.T
    y_cval = reshape_labels(y_cval)
    y_cval = y_cval.T.reshape(10, X_cval.shape[1])

    parameters = model(X_train, y_train, n_hl=85, num_iterations=1750, print_cost=True)
    predictions = predict(parameters, X_test)

    print('Test set accuracy: ', np.mean(predictions == y_test) * 100, '%')

    #Plot images
    # plt.imshow(X_train[5768].reshape(28,28))
    # plt.show()
    # print(y_train[5768])


if __name__ == '__main__':
    main()