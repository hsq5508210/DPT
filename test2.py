import IO
import fileOp
import numpy as np
import washData as wd
from tqdm import tqdm
np.set_printoptions(threshold = np.inf)
def process(path, filename, num):
    data = IO.readData(path+filename)
    timeCol = data[:, 3:4]
    data = np.delete(data, 3, 1)
    onehot = wd.processTime(timeCol, num = 6)
    data = np.concatenate((data, onehot), axis=1)
    IO.writeData(path, data, "trainEncodeTime"+str(num))
def trans2bool(path, filename, num):
    dataDis = IO.readData(path+filename)
    dataDis[:, 4:] = dataDis[:, 4:].astype(np.bool)
    IO.writeData("/data2/lt/ctr/train/minilizeData/", dataDis, "discereUser_info"+str(num))

path = "/data2/lt/ctr/train/disceretData/userInfo/"
path2 = "/data2/lt/ctr/train/minilizeData/"
trainDataName = fileOp.file_name(path)
readName = fileOp.file_name(path2)
for i in range(5):
    # trans2bool(path, trainDataName[0][4], 4)
    d = IO.readData(path2+readName[0][i])
    print(d[1])





