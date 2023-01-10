import os
import sys
import numpy as np
from spde import spde


def fitPar(model,data,start):
    vers = findFits(model,data,start)
    res = True
    for i in range(vers.shape[0]):
        print(str(i)+" of " + str(vers.shape[0]))
        mod = spde(model = model)
        res = mod.fitTo(data,vers[i,1],vers[i,2], vers[i,0],verbose = True)
        if not res:
            print(str(vers[i,1])+" failed")
    return(res)

def findFits(model, data,start):
    vers = np.array([[i,j,k] for i in range(start,start + 5) for j in range(1,4) for k in range(1,4)])
    modstr = np.array(["SI", "SA", "NI","NA"])
    dho = np.array(["100","10000","27000"])
    r = np.array(["1","10","100"])
    for file in os.listdir("./fits/"):
        if file.startswith(modstr[model-1]+"-"+modstr[data-1]):
            tmp = file.split('-')[2:]
            tdho = np.where(dho == tmp[0][3:])[0][0] + 1
            tnum = int(tmp[2].split(".")[0])
            tr = np.where(r==tmp[1][1:])[0][0]+1
            tar = np.array([tnum,tdho,tr]) # num , dho , r
            vers = np.delete(vers,np.where((vers == tar).all(axis=1))[0],axis=0)
    return(vers)

def main(argv):
    start = int((int(argv[0])-1)*5 + 1)
    res = fitPar(4,4,start)
    return(res)


if __name__ == "__main__":
   main(sys.argv[1:])