import IO 
import fileOp
import washData as wd
import  numpy as np
import pandas as pd
# Print the detial of the data.
# np.set_printoptions(threshold=np.inf)
import IO 
import fileOp
import washData as wd
import  numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
# Print the detial of the data.
# np.set_printoptions(threshold=np.inf)
def process(path, name, markData, num):
    data = IO.readData(path+str(name))
    print("Before wash, the shape is: ", data.shape)
    # print(data[1:4, :])
    data = wd.delMarkedData(data, markData, 1)
    print("After wash, the shape is: ", data.shape)
    IO.writeData("/home/wendy/results/hsq/2ndClean/", data, '2ndTrainCleand_' + str(num))


path = "/home/wendy/results/hsq/cleaned/"
pathmusr = "/home/wendy/results/hsq/MarkedData/usrinfo_mark.npy"
usrMarked = wd.getMarked(IO.readData(pathmusr, 'npy'))
usrMarked = np.reshape(usrMarked, [532648, 1])
trainDataName = fileOp.file_name(path)
usrMarked = fileOp.splitData(data = usrMarked, num = 256)
executor = ThreadPoolExecutor(max_workers = 8)
task = []
for i in range(2, 18):
    print("processing...")
    # print(usrMarked[i][1:4, :])

    process(path, trainDataName[0][i], usrMarked[1], i+1)
