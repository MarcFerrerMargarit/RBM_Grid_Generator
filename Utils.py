#Return the max lenght that each vector must have
def getMaxLengthForElements(vector):
    maxLength = 0
    for i in range(len(vector)):
        tmp = vector[i]
        for j in range(len(tmp)):
            if(len(tmp[j])>maxLength):
                maxLength = len(tmp[j])
    return maxLength

#Return a vector which contains the transformed input data
def transformInputVector(vector, maxLength, length):
    output = []
    for i in range(length):
        v = vector[i]
        for j in range(maxLength):
            tmp = []
            length_v = len(v)
            for k in range(length_v):
                if(j >= len(v[k])):
                    tmp.append(0)
                else:
                    tmp.append(v[k][j])
            output.append(tmp)
    return output

#Clean duplicate elements of the output data
def cleaningOutput(vector):
	final_output_grid = []
	for i in range(len(vector)):
		tmp = list(set(vector[i]))
		tmp.sort()
		final_output_grid.append(tmp)
	return final_output_grid