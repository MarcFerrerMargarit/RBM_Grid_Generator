import pandas as pd
import json
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent =  currentdir + '\RBM'
sys.path.insert(0,parent)
from timeit import default_timer as timer
import time
import matplotlib.pyplot as plt
import numexpr  as ne
import profile
import pandas
from random import randint
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import BernoulliRBM
import Utils
import pickle

if __name__ == '__main__':
    vector_data = pd.read_json('./Transformed_Moodboards.json')
    A = vector_data.values
    data_modern = []
    A = A.ravel()
    length_vectors = len(A)
    for i in range(A.shape[0]):
        len_v = len(A[i])
        tmp = []
        for j in range(len_v):
            tmp.append(A[i][j])
        tmp = np.asarray(tmp).ravel()
        data_modern.append(tmp)
    data_modern = np.asarray(data_modern)
    input_data = []
    input_data.append(data_modern)
    input_data = np.asarray(input_data)
    max_elements = Utils.getMaxLengthForElements(input_data[0])
    f_data = Utils.transformInputVector(input_data[0], max_elements, length_vectors)

    # Multiply data for obtain more range <= 10000.
    for i in range(50):
        for k in range((length_vectors * max_elements)):
            f_data.append(f_data[k])
    f_data = np.asarray(f_data)
    print(f_data.shape)
    oneHotEncoder = OneHotEncoder(262, sparse=True).fit(f_data)
    oneHotData = oneHotEncoder.transform(f_data)
    RBM_Machine = BernoulliRBM(n_components=262, learning_rate=0.01, n_iter=50, random_state=0, verbose=True)
    # RBM_Machine = BernoulliRBM(n_components=262, learning_rate=0.01,batch_size=64, n_iter=20, random_state=0, verbose=True)
    RBM_Machine.fit(oneHotData)
    filename = 'RBM.pickle'
    pickle.dump(RBM_Machine, open(filename, 'wb'))
    pickle_out = open("OneHotData.pickle", "wb")
    pickle.dump(oneHotData, pickle_out)
    pickle_out.close()
