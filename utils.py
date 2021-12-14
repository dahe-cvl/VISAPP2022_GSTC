import os
import numpy as np

def createFolder(path):
    if not os.path.exists(path):
        print("create folder " + str(path))
        os.mkdir(path)
    else:
        print("folder already exsists - [" + str(path) + "]")

def csvWriter(dst_folder="", name="metrics_history.log", entries_list=None):
    if (entries_list == None):
        print("ERROR: entries_list must have a valid entry!")

    # prepare entry_line
    entry_line = ""
    for i in range(0, len(entries_list)):
        tmp = entries_list[i]
        entry_line = entry_line + ";" + str(tmp)

    fp = open(dst_folder + "/" + str(name), 'a')
    fp.write(entry_line + "\n")
    fp.close()

def csvReader(name="metrics_history.log"):
    #print(name)
    fp = open(name, 'r')
    lines = fp.readlines()
    fp.close()

    entries_l = []
    for line in lines:
        line = line.replace('\n', '')
        line = line.replace('', '')
        #print(line)
        line_split = line.split(';')

        tmp_l = []
        for split in line_split[:-1]:
            tmp_l.append(split)
        #print(tmp_l)
        entries_l.append(tmp_l)

    entries_np = np.array(entries_l)
    return entries_np