import washData as wd
import IO
import fileOp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import  SourceModule

def process(path, name, markData):
    data = IO.readData(path+str(name))
    data = wd.delMarkedData(data, markData, 1)
    IO.writeData("/data2/lt/ctr/train/cleaned/", data, str(name) + "_cleand")

path = "/data2/lt/ctr/train/batch/"
pathmusr = "/data2/lt/ctr/MarkedData/usrinfo_mark.npy"
usrMarked = wd.getMarked(IO.readData(pathmusr, 'npy'))
usrMarked = np.reshape(usrMarked, [532648, 1])
trainDataName = fileOp.file_name(path)
usrMarked = fileOp.splitData(data = usrMarked, num = 256)
# m = trainDataName.shape[0]
# print(len(usrMarked))
# mark = usrMarked[0:2000]
# test1 = process(path, trainDataName[0][0], usrMarked)
# print(usrMarked[1].shape)
# process(path, trainDataName[0][1], usrMarked[1])
#=======================================================
executor = ThreadPoolExecutor(max_workers = 28)
task = []
for i in range(28):
    t = executor.submit(process(path, trainDataName[0][i], usrMarked[0]))
    # print(usrMarked[i])
    # t = executor.submit(process(path, trainDataName[0][1], usrMarked[i]))
    task.append(t)
# for i in range(28):






