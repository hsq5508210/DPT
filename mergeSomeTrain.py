from DPT import *
from tqdm import tqdm
import pandas as pd
import cudf
# userInfo = IO.readData()
def merge():
    trainpath = "/data2/lt/ctr/train/disceretData/batchTrain/"
    trainNames = fileOp.file_name(trainpath)
    trainSets = []
    for i in range(28):
        trainSets.append(IO.readData(trainpath+str(trainNames[0][i])))
    # Merge trainset.
    TRAIN = trainSets[0]
    for i in tqdm(range(1, 28)):
        TRAIN = np.vstack([TRAIN, trainSets[i]])
        print(TRAIN)

    IO.writeData("/data2/lt/ctr/train/disceretData/", np.array(TRAIN), "mergeTRAIN")
    print(TRAIN)
data = IO.readData("/data2/lt/ctr/train/disceretData/batchTrain/trainDisceret_0.npy")
# print(data)
# userInfo = []
# for i in range(5):
#     userInfo.append(IO.readData("/data2/lt/ctr/train/disceretData/userInfo/minilizeData/"+"discereUser_info"+str(i)+".npy"))
# userInfo = np.array(userInfo)
# for i in range(5):
#     for j in tqdm(range(userInfo[i].shape[0])):
#         map = []
#         map.append(userInfo[i][j][0])
#         map.append(j)
#         MAP.append(map)
MAP = IO.readData("/data2/lt/ctr/train/disceretData/uesrID.npy")
for i in tqdm(range(data.shape[0])):
    m = np.argwhere(MAP[:, 0] == data[i][1])
    # print(data[i], m, MAP[m[0][0]])
    data[i][1] = m[0][0]





