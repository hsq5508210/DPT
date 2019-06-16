import IO 

path1 = "/data2/lt/ctr/train/trainSet.npy"
path2 = "/data2/lt/ctr/train/user_info.npy"
path3 = "/data2/lt/ctr/train/ad_info.npy"
path4 = "/data2/lt/ctr/train/content_info.npy"
traindata = IO.readData(path1, "npy")
print(traindata.shape)
print(traindata)
userinfo = IO.readData(path2, "npy")
print(userinfo.shape)
print(userinfo)
adinfo = IO.readData(path3, "npy")
print(adinfo.shape)
print(adinfo)
content = IO.readData(path4, "npy")
print(content.shape)
print(content)

userId = pd.DataFrame(traindata).iloc[:,1:2]
print("user id is:\n", userId)
