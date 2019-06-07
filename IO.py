import DataProcessTool.DPT
import csv 
import numpy as np 
import pandas as pd 
import time 
def readData(path, fileType = 'csv'):
    print("reading...")
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
    print("Done")
    t_e = str(time.time() - t_s) 
    print("spend " + t_e +'s')
    print("shape is:", data.shape)
    return data
    #save the npmat type data 
def writeData(path, data, fileName, fileType = 'bin'):
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
    print("Done")
    t_e = str(time.time() - t_s)
    print("spend " + t_e +'s')
    del data 
