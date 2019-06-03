import IO 

path = "/Users/sqh/Projects/2019_520_compete/5_20_data/trainset/train_20190518.csv"
data = IO.readData(path)
print(data)
print(data.shape)
IO.writeData("/Users/sqh/Projects/2019_520_compete/5_20_data", data, "trainSet", "npy")
