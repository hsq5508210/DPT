import csv
import numpy as np 
import pandas as pd 
import time
#===================================================
# Read data from different formats.
def readData(path):
    # readData("your file path", "your format")
    #
    #
    print("reading...")
    s = path
    n = len(s)
    # Get the format.
    for i in range(n-1, 1, -1):
        if s[i] == '.':
            s = s[i + 1:]
            break
    fileType = s
    t_s = time.time()
    if fileType == "csv":
        csvfile = pd.read_csv(path)
        data = csvfile
    if fileType == "bin":
        data = np.fromfile(path)
    if fileType == "npy":
        data = np.load(path)
    if fileType == "h5":
        data = pd.read_hdf(path, key = 'data') 
    t_e = str(time.time() - t_s)
    print("spend " + t_e +'s')
    print("shape is:", data.shape)
    return np.array(data)
#=====================================================
# Save the npmat type data.
def writeData(path, data, fileName, fileType = 'npy'):
    # data必须是np.array格式
    # 最好存储为npy格式
    # writeData("/data2/lt/ctr/", numpy_data, "newdata", 'npy')
    print("writing...")
    t_s = time.time()
    path = path+fileName+'.'+fileType 
    if fileType == 'bin':
        data.tofile(path)
    if fileType == "npy":
        np.save(path, data)
    if fileType == "h5":
        h5 = pd.HDFStore(path, 'w')
        h5['data'] = data
        h5.close()
    if fileType == 'csv':
        data.to_csv(path)
    print("Done")
    t_e = str(time.time() - t_s)
    print("spend " + t_e +'s')
    del data 
