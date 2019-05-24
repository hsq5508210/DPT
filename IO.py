import DPT
import csv 
import numpy as np 

def readData(path, fileType = 'csv'):
    if fileType == "csv":
        csvfile = np.loadtxt(path,  dtype=np.str, delimiter=",")
        data = csvfile
    if fileType == "bin":
        data = np.fromfile(path)
    if fileType == "npy":
        data = np.load(path)
    print("Done")
    return data
    #save the npmat type data 
def writeData(path, data, fileName, fileType = 'bin'):
    path = path+fileName+'.'+fileType 
    if fileType == 'bin':
        data.tofile(path)
    if fileType == "npy":
        np.save(path, data)
    print("Done")
