import IO

path = "/data2/lt/ctr/content_info/content_info.csv"
data = readData(path, fileType = 'csv')
print(data.shape)
writeData("/data2/lt/ctr/train/", data, "content_info", fileType = 'npy')
