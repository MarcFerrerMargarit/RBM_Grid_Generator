def getMaxLengthForElements(vector):
    maxLength = 0
    for i in range(len(vector)):
        if(len(vector[i]) > maxLength):
            maxLength = len(vector[i])
    return maxLength

def transformInputVector(vector, maxLength):
    output_data = []
    for i in range(maxLength):
        tmp = []
        for j in range(len(vector)):
            if(i < len(vector[j])):
                tmp.append(vector[j][i])
            else:
                tmp.append(0)
        output_data.append(tmp)
    return output_data

def generate_data_fake(): 
    vec = [] 
    for i in range(1000): 
        tmp = [] 
        for k in range(48): 
            tmp.append(0) 
        vec.append(tmp) 
    return np.asarray(vec) 
                
if __name__ == "__main__":
    testVector = [[1],[1,2,3],[3,4,3,3]]
	print getMaxLengthForElements(testVector)