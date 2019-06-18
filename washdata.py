import IO
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
import time

#====================================
# read the data
print("Reading all the data.....")
path1 = "/data2/lt/ctr/train/trainSet.npy"
path2 = "/data2/lt/ctr/train/user_info.npy"
path3 = "/data2/lt/ctr/train/adInfo_1.npy"
path4 = "/data2/lt/ctr/train/content_info.npy"

# 17 dimensions.
traincolumns = ['label', 'uId', 'adId', 'operTime', 'siteId', 'slotId', 'contentId', 'netType']
uinfcolumns = ['uId', 'age', 'gender', 'city', 'province', 'phoneType', 'carrier']
adinfcolumns = ['adId', 'billId', 'primId', 'creativeType', 'interType', 'spreadAppId']
contentcolumns = ['contentId', 'firstClass', 'secondClass']


# Stack all the index key and paths.
columns = [traincolumns, uinfcolumns, adinfcolumns, contentcolumns]
path = [path1, path2, path3, path4]
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
# Import the data.
# data[0] -> traindata
# data[1] -> userInfo
# data[2] -> adInfo
# data[3] -> content
data = [1, 2 , 3, 4]
for i in range(4):
    if i == 0: continue
    data[i] = (importData(path[i], columns[i]))
print("Already read all the data!")
#    print(data[i])
for i in range(4):
    if i == 0 or i == 3: continue
    data[i] = (filterOfNan(data[i]))

traindata = data[0]
userInfo = data[1]
adInfo = data[2]
content = data[3]

IO.writeData('/data2/lt/ctr/MarkedData/', pd.DataFrame(userInfo), 'userInfo', 'h5')
IO.writeData('/data2/lt/ctr/MarkedData/', pd.DataFrame(adInfo), 'adInfo', 'h5')



#===========================================



#===========================================







# Clean the memory.
exit()


