import IO 

path1 = "/Users/sqh/Projects/2019_520_compete/5_20_data/test_data/train_demo.csv"
path2 = "/Users/sqh/Projects/2019_520_compete/5_20_data/test_data/userInfo_demo.csv"
path3 = "/Users/sqh/Projects/2019_520_compete/5_20_data/test_data/adInfo_demo.csv"
path4 = "/Users/sqh/Projects/2019_520_compete/5_20_data/test_data/contentInfo_demo.csv"
traindata = IO.readData(path1, "csv")
print(traindata.shape)
print(traindata)
userinfo = IO.readData(path2, "csv")
print(userinfo.shape)
print(userinfo)
adinfo = IO.readData(path3, "csv")
print(adinfo.shape)
print(adinfo)
content = IO.readData(path4, "csv")
print(content.shape)
print(content)

userId = traindata.iloc[:,1:2]
print("user id is:\n",userId)
