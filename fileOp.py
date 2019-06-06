import DataProcessTool.IO 
import os 

def file_name(path):
    file = []
    for files in os.walk(path):
        file.append(files)
    return file

