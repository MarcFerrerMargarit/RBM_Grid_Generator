import pickle
import random
import numpy as np


# Return the max length that each vector must have
def getMaxLengthForElements(vector):
    maxLength = 0
    for i in range(len(vector)):
        tmp = vector[i]
        for j in range(len(tmp)):
            if (len(tmp[j]) > maxLength):
                maxLength = len(tmp[j])
    return maxLength


# Return a vector which contains the transformed input data
def transformInputVector(vector, maxLength, length):
    output = []
    for i in range(length):
        v = vector[i]
        for j in range(maxLength):
            tmp = []
            length_v = len(v)
            for k in range(length_v):
                if j >= len(v[k]):
                    tmp.append(0)
                else:
                    tmp.append(v[k][j])
            output.append(tmp)
    return output


# Clean duplicate elements of the output data
def cleaningOutput(vector):
    final_output_grid = []
    for i in range(len(vector)):
        tmp = list(set(vector[i]))
        tmp.sort()
        final_output_grid.append(tmp)
    return final_output_grid


# Generate grid from data
def generateGrid(number_grids, rbm_name="", onehot_name=""):
    pickle_in = open(onehot_name, "rb")
    data = pickle.load(pickle_in)
    pickle_in_rbm = open(rbm_name, "rb")
    RBM = pickle.load(pickle_in_rbm)
    all_grids = []
    for j in range(number_grids):
        final_output = []
        for i in range(50):
            x_visible = RBM.gibbs(data[random.randint(0, data.shape[0])])
            x_visible = x_visible.ravel()
            final_vector = []
            for k in range(len(x_visible)):
                if x_visible[k] == True:
                    final_vector.append(1)
                else:
                    final_vector.append(0)
            final_Data = []
            for i in range(48):
                tmp = final_vector[(262 * i):((i + 1) * 262)]
                if 1 not in tmp:
                    index = 0
                else:
                    index = tmp.index(1)
                final_Data.append(index)
            final_output.append(final_Data)
        final_output = np.asarray(final_output)
        final_data_output = []
        for k in range(48):
            t = []
            for i in range(len(final_output)):
                t.append(final_output[i][k])
            final_data_output.append(t)
        final_data_output = np.asarray(final_data_output)
        output_grid = []
        for i in range(48):
            v = final_data_output[i].tolist()
            v_filter = list(filter((0).__ne__, v))
            if len(v_filter) == 0:
                v_filter.append(0)
            output_grid.append(v_filter)
        all_grids.append(cleaningOutput(output_grid))
    return all_grids
