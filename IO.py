import DPT
import csv

class IO(object):
    def __init__():
    
    def readData(path, fileType = csv):
        if fileType == "csv":
            csvfile = np.loadtxt(path,  dtype=np.str, delimiter=",")
            data = csvfile
            return data
        #else:
