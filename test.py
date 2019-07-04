import IO 
import fileOp
import washData as wd
import  numpy as np
import pandas as pd
# Print the detial of the data.
# np.set_printoptions(threshold=np.inf)

path1 = "/data2/lt/ctr/train/npy/trainSet.npy"
path2 = "/data2/lt/ctr/train/npy/user_info.npy"
path3 = "/data2/lt/ctr/train/npy/ad_info.npy"
path4 = "/data2/lt/ctr/train/npy/newContent_info.npy"
pathmad = "/data2/lt/ctr/MarkedData/adinfo_mark.npy"
pathmusr = "/data2/lt/ctr/MarkedData/usrinfo_mark.npy"

# trainData = IO.readData(path1, 'npy')
# pathwrite = "/data2/lt/ctr/train/batch/"
# train = []
# train = fileOp.splitData(path1, num = 56)
# m = len(train)
# for i in range(m):
#     IO.writeData(pathwrite, train[i], "train_batch"+str(i), 'npy')
#     print(train[i].shape)
usrMarked = wd.getMarked(wd.filterOfNan(IO.readData(path2, 'npy')))
print(len(usrMarked))
# adMarked = IO.readData(pathmad, 'npy')
# # print(adMarked)
# adMarked = wd.getMarked(adMarked)
# print(trainData)
# data = wd.delMarkedData(trainData, usrMarked, 1)
# print(len(usrMarked), '\n', len(adMarked))
exit()









