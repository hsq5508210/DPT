import IO
import numpy as np
np.set_printoptions(threshold = np.inf)
path = "/data2/lt/ctr/train/cleaned/train_batch1.npy_cleand.npy"
data = IO.readData(path)
for i in range(data.shape[0]):
    if data[i][1] == 1:
        print(data[i])
