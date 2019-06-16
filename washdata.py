import IO
import pandas as pd
#-------------------------------------
# read the data
path1 = "/data2/lt/ctr/train/trainSet.npy"
path2 = "/data2/lt/ctr/train/user_info.npy"
path3 = "/data2/lt/ctr/train/ad_info.npy"
path4 = "/data2/lt/ctr/train/content_info.npy"
traindata = IO.readData(path1, "npy")
userinfo = IO.readData(path2, "npy")
adinfo = IO.readData(path3, "npy")
content = IO.readData(path4, "npy")
print("Already read all the data!")
userId = pd.DataFrame(traindata).iloc[:,1:2]

#-------------------------------------------

# generate a hash map table from "userId"
userId["hash"] = 0
print("Generate userId-hash complete!")



# Clean the memory.
exit()

