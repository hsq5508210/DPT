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
path3 = "/data2/lt/ctr/train/ad_info.npy"
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
# Import the data.
# data[0] -> traindata
# data[1] -> userInfo
# data[2] -> adInfo
# data[3] -> content
data = []
# for i in range(4):
#     data.append(importData(path[i], columns[i]))
#
# traindata = data[0]
# userInfo = data[1]
# adInfo = data[2]
# content = data[3]
userInfo = importData(path[1], columns[1])
print("Already read all the data!")

# generate a hash map table from "userId"
userId = pd.DataFrame(userInfo.iloc[:, 0:1])
print("Generating the userId-hash..... ")
userId["hash"] = 0
userInfo["hash"] = 0
print("Generate userId-hash complete!")
#===========================================

# Index the "NULL" data and mark the hash map.
usrIfoNan = userInfo[userInfo.isnull().values == True]
m = usrIfoNan.shape[0]
usrIfoNan = np.array(usrIfoNan)
usrHash = np.array(userId)
#print(usrHash)
print(usrIfoNan.shape)
for i in tqdm(range(m)):
# for i in range(m):
    for j in range(1, 7):
        #print(usrIfoNan[i][j])
        if np.isnan(usrIfoNan[i][j]) == True:
            usrIfoNan[i][7] += 1
print(usrIfoNan)

# Stage the usrIfoNan.
IO.writeData('/data2/lt/ctr/train/', np.array(usrIfoNan), 'userIdHash', 'npy')




# Clean the memory.
exit()


