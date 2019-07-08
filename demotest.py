# -*- coding:UTF-8 -*-
import washData as wd
import numpy as np
import IO
np.set_printoptions(threshold=np.inf)

# path = "/home/wendy/results/hsq/npy/newContent_info.npy"
# origin_data = IO.readData(path)
# print(origin_data)
# dataCol = origin_data[:, 1:]
# dis_data = wd.val2DiscreteArr(dataCol)
# id = origin_data[:, 0:1]
# print id.shape
# print dis_data.shape
# data = np.concatenate((id, dis_data),axis=1)
# print data
# IO.writeData("/home/wendy/results/hsq/npy/", dataCol, 'discereContent_info')
# print(dataCol.shape)

# path = "/home/wendy/results/hsq/npy/ad_info.npy"
# origin_data = IO.readData(path)
# # print(origin_data)
# creativeType = origin_data[:, 3:4]
# intertype = origin_data[:, 4:5]
# cols = [creativeType, intertype]
# retdata = []
# for i in range(2):
#     #print cols[i].shape
#     d = wd.val2DiscreteArr(cols[i])
#     print d.shape
#     origin_data = np.concatenate((origin_data, d), axis=1)
#
# origin_data = np.delete(origin_data, 4, 1)
# origin_data = np.delete(origin_data, 3, 1)
# print origin_data
# IO.writeData("/home/wendy/results/hsq/npy/", origin_data, 'discereAd_info')
# print(dataCol.shape)

path = "/home/wendy/results/hsq/npy/user_info.npy"
origin_data = IO.readData(path)
print(origin_data[1:4, :])
gender = origin_data[:, 2:3]
city = origin_data[:, 3:4]
province = origin_data[:, 4:5]
phoneType = origin_data[:, 5:6]
carrier = origin_data[:, 6:7]
cols = [gender, city, province, phoneType, carrier]
origin_data = np.delete(origin_data, [2,3,4,5,6], 1)
print origin_data[1:4, :]
# for i in range(5):
#print cols[i].shape
d = wd.val2DiscreteArr(cols[1])
print d.shape
IO.writeData("/home/wendy/results/hsq/npy/", d, 'discereUser_info_'+str(1))

# origin_data = np.concatenate((origin_data, d), axis=1)
# print origin_data[1:4, :]
# IO.writeData("/home/wendy/results/hsq/npy/", origin_data, 'discereUser_info')


