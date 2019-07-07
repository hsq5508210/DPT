import IO
import os 
import numpy as np
#================================================
# Get file names in a dir 'path'.
def file_name(path):
    f = []
    for root, dirs, file in os.walk(path):
        f.append(file)
    return list(f)
#================================================
# Split the dataSet
def splitData(path = None, data = None, dataType = "npy", num = 10):
    if path != None:
        data = IO.readData(path, dataType)
        # Feed the np.array data.
        print("Spliting...")
        if dataType != "npy":
            data = np.array(data)
    retdata = []
    sumLen = data.shape[0]
    print("shape is:", data.shape)
    subLen = int(sumLen/num)
    cnt = 0
    for i in range(num):
        if i == num-1:
            retdata.append(np.array(data[cnt:sumLen-1, :]))
        else:
            retdata.append(np.array(data[cnt:cnt+subLen, :]))
        cnt += subLen
    return retdata
#================================================
