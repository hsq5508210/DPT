import IO
import numpy as np
from tqdm import tqdm
# np.set_printoptions(threshold = np.inf)
def clean(path):
    data = IO.readData(path)
    print("Before clean, the shape is: ", data.shape)
    retdata = []
    for i in tqdm(range(data.shape[0])):
        if data[i][1] != 1:
            retdata.append(data[i])
        else: print(data[i])
    retdata = np.array(retdata)
    print("After clean, the shape is: ", retdata.shape)
    return retdata
print(clean("/data2/lt/ctr/train/cleaned/train_batch0.npy_cleand.npy"))


