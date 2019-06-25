import IO
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time

#====================================
#=====================================
def importData(path, columns, type = 'npy'):
    data = IO.readData(path, type)
    data = pd.DataFrame(data)
    data.columns = columns
    return data
#=====================================
# Index the "NULL" data and mark the hash map.
def filterOfNan(matrix):
    # Matrix is pandas dataframe format!!!.
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
# Get the marked items.
def getMarked(data):
    data = np.array(data)
    m = data.shape[0]
    # Get the Hash column.
    n = data.shape[1] - 1
    marked = []
    for i in tqdm(range(m)):
        if data[i][n] == 1:
            # Append the id of item.
            marked.append(data[i][0])
    # Return the pylist.
    return marked
#========================================
# Delete the marked items.
def delMarkedData(data, markedIndex, col):
    data = np.array(data)
    retData = []
    markedIndex = np.array(markedIndex)
    m = data.shape[0]
    n = markedIndex.shape[0]
    for i in tqdm(range(m)):
        for j in range(n):
            if data[i][col] == markedIndex[j]:
                np.delete(data, i, 0)
    return data
#========================================