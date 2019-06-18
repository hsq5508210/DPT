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
    return matrixNan
#===========================================
