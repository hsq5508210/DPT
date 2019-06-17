import IO
import pandas as pd
#-------------------------------------
# read the data
print("Reading all the data.....")
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
print("Generating the userId-hash..... ")
userId["hash"] = 0
print("Generate userId-hash complete!")

# Transform format of the all data.
print("Transforming the data.....")
train_pd = (pd.DataFrame(traindata)).reindex(columns = ['label', 'uId', 'adId', 'operTime', 'siteId', 'slotId', 'contentId', 'netType'])
del traindata
userinfo_pd = (pd.DataFrame(userinfo)).reindex(columns = ['uId', 'age', 'gender', 'city', 'province', 'phoneType', 'carrier'])
del userinfo
content_pd = (pd.DataFrame(content)).reindex(columns = ['contentId', 'firstClass', 'secondClass'])
del content
adinfo_pd = (pd.DataFrame(adinfo)).reindex(colunms = ['adId', 'billId', 'primId', 'creativeType', 'interType', 'spreadAppId'])
del adinfo
print("Transformed all the data!")

# Index the "NULL" data and mark the hash map.
print(userinfo_pd.columns)

# Clean the memory.
exit()

