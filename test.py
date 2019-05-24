import IO 

path = "/Users/sqh/Projects/DataProcessTool/data.csv"
data = IO.readData(path)
print(data)
print(data.shape)
IO.writeData("/Users/sqh/Projects/DataProcessTool/", data, "data", "npy")
data = IO.readData("/Users/sqh/Projects/DataProcessTool/data.npy", "npy")
print(data.shape)
print(data)
