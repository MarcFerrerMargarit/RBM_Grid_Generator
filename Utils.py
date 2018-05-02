def getMaxLengthForElements(vector):
    maxLength = 0
    for i in range(len(vector)):
        tmp = vector[i]
        for j in range(len(tmp)):
            if(len(tmp[j])>maxLength):
                maxLength = len(tmp[j])
    return maxLength

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