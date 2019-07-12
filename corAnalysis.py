from DPT import *
import time
from numba import *
from tqdm import tqdm
import computes
trainCode = IO.readData("/data2/lt/ctr/train/npy/code_train.npy")
label = trainCode[:, 0]
trainCode = np.delete(trainCode, 0, 1)
# print(label[0:4])
print(trainCode.shape)

@jit
def getCorAna():
    retList = []
    for i in tqdm(range(trainCode.shape[1])):
        line = []
        data = trainCode[:, i]
        ca = computes.Pierson(data, label)
        # del data
        line.append(i+1)
        line.append(ca)
        retList.append(line)

    retList = np.array(retList, dtype="str")
    IO.writeData("/data2/lt/ctr/train/npy/", retList, "corAna")

getCorAna()


