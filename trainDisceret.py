import IO
import washData as wd
import numpy as np
import fileOp

def process(path, name, num):
    data = IO.readData(path+str(name))
    timeCol = data[:, 3:4]
    netTypeCol = data[:, 7:]
    data = np.delete(data, [3, 7], 1) # Delete the time and netType col.
    timeCol = wd.processTime(timeCol, 6) # 生成6个维度
    netTypeCol = wd.val2DiscreteArr(netTypeCol) # 生成6个维度
    data = np.concatenate((data,timeCol,netTypeCol), axis=1) # time前netType后。
    IO.writeData("/data2/lt/ctr/train/disceretData/batchTrain/", data, 'trainDisceret_'+str(i))
    # data = wd.val2DiscreteArr()


path = "/data2/lt/ctr/train/batch/"
trainDataName = fileOp.file_name(path)
for i in range(18, 57):
    process(path, trainDataName[0][i], i)
    print("It's batch NO."+str(i))