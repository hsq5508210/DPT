import IO
import pandas as pd
import washData as wd

#====================================
# read the data
print("Reading all the data.....")
path1 = "/data2/lt/ctr/train/trainSet.npy"
path2 = "/data2/lt/ctr/train/user_info.npy"
path3 = "/data2/lt/ctr/train/adInfo_1.npy"
path4 = "/data2/lt/ctr/train/content_info.npy"

# 17 dimensions.
traincolumns = ['label', 'uId', 'adId', 'operTime', 'siteId', 'slotId', 'contentId', 'netType']
uinfcolumns = ['uId', 'age', 'gender', 'city', 'province', 'phoneType', 'carrier']
adinfcolumns = ['adId', 'billId', 'primId', 'creativeType', 'interType', 'spreadAppId']
contentcolumns = ['contentId', 'firstClass', 'secondClass']


# Stack all the index key and paths.
columns = [traincolumns, uinfcolumns, adinfcolumns, contentcolumns]
path = [path1, path2, path3, path4]

# Import the data.
# data[0] -> traindata
# data[1] -> userInfo
# data[2] -> adInfo
# data[3] -> content
data = [1, 2 , 3, 4]
for i in range(4):
    if i == 0: continue
    data[i] = (wd.importData(path[i], columns[i]))
print("Already read all the data!")
#    print(data[i])
for i in range(4):
    if i == 0 or i == 3: continue
    data[i] = (wd.filterOfNan(data[i]))

traindata = data[0]
userInfo = data[1]
adInfo = data[2]
content = data[3]

IO.writeData('/data2/lt/ctr/MarkedData/', pd.DataFrame(userInfo), 'userInfo', 'h5')
IO.writeData('/data2/lt/ctr/MarkedData/', pd.DataFrame(adInfo), 'adInfo', 'h5')



#===========================================



#===========================================







# Clean the memory.
exit()


