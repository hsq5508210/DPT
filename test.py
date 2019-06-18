import IO 
import fileOp


path1 = "/data2/lt/ctr/train/csv/train_20190518.csv"
path2 = "/data2/lt/ctr/train/csv/user_info.csv"
path3 = "/data2/lt/ctr/train/csv/ad_info.csv"
path4 = "/data2/lt/ctr/train/csv/content_info.csv"
path = "/data2/lt/ctr/train/h5/"
traindata = IO.readData(path1, "csv")
IO.writeData(path, traindata, 'trainSet', 'h5')
del traindata
userinfo = IO.readData(path2, "csv")
IO.writeData(path, userinfo, 'userInfo', 'h5')
del userinfo
adinfo = IO.readData(path3, "csv")
IO.writeData(path, adinfo, 'adInfo', 'h5')
del adinfo
content = IO.readData(path4, "csv")
IO.writeData(path, content, 'contentInfo', 'h5')
del content
exit()


