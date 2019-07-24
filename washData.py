import IO
import math
import numba
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time

#====================================
#=====================================
def importData(path, columns):
    data = IO.readData(path, type)
    data = pd.DataFrame(data)
    data.columns = columns
    return data
#=====================================
# Index the "NULL" data and mark the hash map.
def filterOfNan(matrix):
    # Matrix is pandas dataframe format!!!.
    matrix = pd.DataFrame(matrix)
    matrixNan = matrix[matrix.isnull().values == True]
    # Generate hash column.
    matrixNan['hash'] = 0
    matrixNan = np.array(matrixNan)
    m = matrixNan.shape[0]
    print(matrixNan.shape)
    n = matrixNan.shape[1]
    o = 1
    for i in tqdm(range(m)):
        cnt = 0
        for j in range(o, n-1):
            if type(matrixNan[i][j]) == 'str':
                continue
            if np.isnan(matrixNan[i][j]) == True:
                cnt += 1
        if cnt >= 2:
            matrixNan[i][n-1] = 1
    # Return the np.array
    return matrixNan
#=========================================
# Statics the class of the data
def encodeTheClass(data, col):
    # 'data' is numpy array format
    # 'col' is index of the column you choose
    data = np.array(data)
    column = data[:, col]
    # print(column)
    # deprive the same class
    column = list(set(column))
    m = data.shape[0]
    n = len(column)
    for i in tqdm(range(m)):
        for j in range(n):
            if data[i][col] == column[j]:
                # encode
                data[i][col] = j
    print(data[:][col])
    return data
#========================================
# Transform the value class to discrete array.
def val2DiscreteArr(dataCol):
    # 'dataCol' is a np.array type column of the data.
    # Static the max value of the data.
    m = dataCol.shape[0]
    n = 0
    for i in range(m):
        if dataCol[i][0] > n:
            n = dataCol[i][0]
    print("The max value is: ", n)
    retdata = np.zeros([m, int(n)])
    # Encode(mark) the dim.
    for i in tqdm(range(m)):
        if np.isnan(dataCol[i][0]) == False:
            retdata[i][int(dataCol[i][0] - 1)] = 1
    return retdata
#========================================
# Get the marked items.
def getMarked(data):
    data = np.array(data)
    m = data.shape[0]
    # Get the Hash column.
    n = data.shape[1] - 1
    marked = []
    print("Getting mark...")
    for i in tqdm(range(m)):
        if data[i][n] == 1:
            # Append the id of item.
            marked.append(data[i][0])
    # Return the np.array.
    return np.array(marked)
#========================================
# Delete the marked items.
def delMarkedData(data, markedIndex, col):
    print("Washing...")
    data = np.array(data)
    markedIndex = np.array(markedIndex)
    n = markedIndex.shape[0]
    dataCol = data[:, col:col+1]
    val = []
    for i in tqdm(range(n)):
        if dataCol[dataCol == markedIndex[i]].size != 0:
            val.append(dataCol[dataCol == markedIndex[i]])
    val = np.array(val)
    v = []
    val = np.array(val).reshape((-1, 1))
    for i in range(val.shape[0]):
        if i == 0: v.append(val[0])
        for j in range(len(v)):
            if val[i] == v[j]:
                break
            if j == len(v)-1:
                v.append(val[i])
    val = np.array(v)
    for i in range(val.shape[0]):
        if i == val.shape[0]: break
        index = np.unique(np.argwhere(data[:, 1] == val[i]))
        data = np.delete(data, index, 0)
    return data

#========================================
# Encode the time information.
def processTime(timeCol, num = 6):
    retdata = timeCol
    m, n = timeCol.shape
    for i in tqdm(range(m)):
        # time[i] = time[i][11:13]
        s = str(timeCol[i])
        timeCol[i] = math.ceil(int(s[13:15])/(24/num))
    retdata = val2DiscreteArr(timeCol)
    return retdata
#========================================