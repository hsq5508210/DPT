import IO

path = "/data2/lt/ctr/ad_info/ad_info.csv"
data = readData(path, fileType = 'csv')
print(data.shape)
writeData("/data2/lt/ctr/train/", data, "ad_info", fileType = 'npy')
