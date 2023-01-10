import sys
import numpy as np
import os
from spde import spde
from scipy import sparse


def main():
    modstr = np.array(["SI", "SA", "NA"])
    dho = np.array(["100","10000"])
    r = np.array(["1","10"])
    totres = np.zeros((9,4))
    for i in range(len(modstr)):
        for j in range(len(dho)):
            for k in range(len(r)):
                res = None
                for l in np.arange(1,101):
                    if os.path.exists("./fits/s2-"+modstr[i] +'-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npy"):
                        if res is None:
                            res = np.load("./fits/s2-"+modstr[i] +'-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npy")
                        else:
                            res = np.vstack([res,np.load("./fits/s2-"+modstr[i] +'-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npy")])
                totres[i*3:(i*3+3),j*2 + k] = np.sqrt((res**2).mean(axis = 0))
    print(totres)

if __name__ == "__main__":
   main()