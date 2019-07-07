import IO 
import fileOp
import washData as wd
import  numpy as np
import pandas as pd
# Print the detial of the data.
# np.set_printoptions(threshold=np.inf)
def process(path, name, markData, num):
    data = IO.readData(path+str(name))
    data = wd.delMarkedData(data, markData, 1)
    IO.writeData("/data2/lt/ctr/train/2ndClean/", data, '2ndTrainCleand_' + str(num))

path = "/data2/lt/ctr/train/cleaned/"
pathmusr = "/data2/lt/ctr/MarkedData/usrinfo_mark.npy"
usrMarked = wd.getMarked(IO.readData(pathmusr, 'npy'))
usrMarked = np.reshape(usrMarked, [532648, 1])
trainDataName = fileOp.file_name(path)
usrMarked = fileOp.splitData(data = usrMarked, num = 256)
for i in range(18):
    process(path, trainDataName[0][i], usrMarked[1], i+1)









