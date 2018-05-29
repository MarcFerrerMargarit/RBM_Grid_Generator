import pandas as pd
import json
import sys
import os
import inspect
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import BernoulliRBM
import Utils
import pickle
from Constants import Constants
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent = currentdir + '\RBM'
sys.path.insert(0, parent)


def main(nameStyle):
    print("Entrenando la màquina. Esto puede tardar varios minutos...")
    vector_data = pd.read_json(Constants.Path_Server_Data + nameStyle + ".json")
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
    oneHotEncoder = OneHotEncoder(262, sparse=True).fit(f_data)
    oneHotData = oneHotEncoder.transform(f_data)
    RBM_Machine = BernoulliRBM(n_components=262, learning_rate=0.01, n_iter=10, random_state=0, verbose=True)
    # RBM_Machine = BernoulliRBM(n_components=262, learning_rate=0.01,batch_size=64, n_iter=20, random_state=0, verbose=True)
    RBM_Machine.fit(oneHotData)
    print("Maquina entrenada")
    filename = Constants.Path_Server_Data + 'RBM' + nameStyle + '.pickle'
    pickle.dump(RBM_Machine, open(filename, 'wb'))
    print("Maquina guardada en: " + filename)
    pickle_out = open(Constants.Path_Server_Data + "OneHotData" + nameStyle + ".pickle", "wb")
    pickle.dump(oneHotData, pickle_out)
    pickle_out.close()
    print("Datos para la màquina guardados en: " + (Constants.Path_Server_Data + "OneHotData" + nameStyle + ".pickle"))


if __name__ == '__main__':
    main("")

