import IO
import os 
import numpy as np
def file_name(path):
    file = []
    for files in os.walk(path):
        file.append(files)
    return file
#================================================
# Split the dataSet
def splitData(path, dataType = "npy", num = 10):
    retdata = []
    data = IO.readData(path, dataType)
    # Feed the np.array data.
    if dataType != "npy":
        data = np.array(data)
    sumLen = data.shape[0]
    subLen = sumLen/num
    cnt = 0
    for i in range(num):
        if i == num-1:
            retdata.append(data[cnt:sumLen-1, :])
        else:
            retdata.append(data[cnt:cnt+subLen, :])
        cnt += subLen
    return retdata
#================================================
