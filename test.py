import IO 
import fileOp
import washData as wd
import  numpy as np
import pandas as pd
# np.set_printoptions(threshold='nan')

# path1 = "/data2/lt/ctr/train/npy/trainSet.npy"
path2 = "/data2/lt/ctr/train/npy/user_info.npy"
path3 = "/data2/lt/ctr/train/npy/ad_info.npy"
path4 = "/data2/lt/ctr/train/npy/newContent_info.npy"
usrinfo = IO.readData(path2, 'npy')
adinfo = IO.readData(path3, 'npy')
usrinfo = pd.DataFrame(usrinfo)
print(usrinfo)
print(adinfo)
adinfo = pd.DataFrame(adinfo)
usrinfo = wd.filterOfNan(usrinfo)
adinfo = wd.filterOfNan(adinfo)
print(usrinfo, '\n', adinfo)
IO.writeData("/data2/lt/ctr/MarkedData/", usrinfo, "usrinfo_mark", 'npy')
IO.writeData("/data2/lt/ctr/MarkedData/", adinfo, "adinfo_mark", 'npy')

# path = "/data2/lt/ctr/train/csv/ad_info.csv"
# traindata = IO.readData(path1, "npy")
# userinfo = IO.readData(path2, "npy")
# adinfo = IO.readData(path, "csv")
# adinfo = wd.encodeTheClass(adinfo, 1)
# IO.writeData("/data2/lt/ctr/train/npy/", adinfo, "ad_info", 'npy')
# print(adinfo)
# content = IO.readData(path4, "npy")
# print(traindata)
# print(userinfo)
# print(content)
exit()







