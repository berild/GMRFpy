
import sys, getopt
import numpy as np
from spde import spde
from joblib import Parallel, delayed
from functools import partial
from datetime import datetime
from tqdm import tqdm
import os


def fit(version,mod,data,vers):
    try: 
        res = mod.fitTo(data,vers[version,1],vers[version,2], vers[version,0],verbose = False)
    except:
        print("Model "+ mod + " data " + data + " dho " + vers[version,1]+ " r " + vers[version,2] + " num " + vers[version,0] + " crashed")
        return(False)
    else:
        return(res)

def fitPar(model,data):
    vers = findFits(model,data)
    mod = spde(model = model)
    fit_ = partial(fit, mod=mod, data = data, vers=vers)
    res = Parallel(n_jobs=20)(delayed(fit_)(i) for i in tqdm(range(vers.shape[0])))
    return(res)

def findFits(model, data):
    vers = np.array([[i,j,k] for i in range(0,100) for j in range(1,4) for k in range(1,4)])
    modstr = ["SI", "SA", "NA"]
    dho = ["100","10000","27000"]
    r = ["1","10","100"]
    mods = []
    print(vers.shape)
    for file in os.listdir("./simulations/"):
        if file.startswith(modstr[model-1]+"-"+modstr[data-1]):
            tmp = file.split('-')[2:]
            tar = np.array([int(tmp.split("-")[2].split(".")[0]),int(np.where(dho == tmp.split("-")[0][3:])[0]) + 1,  int(np.where(r == tmp.split("-")[1][1:])[0])+1 ]) # num , dho , r
            vers = np.delete(vers,np.where((vers == tar).all(axis=1))[0],axis=0)
    print(vers.shape)
    return(vers)

def main(argv):
    modstr = ["Stationary Isotropic", "Stationary Anistropic", "Non-stationary Simple Anisotropic","Non-stationary Complex Anisotropic"]
    mods = None
    if "-" in argv[0]:
        mods = argv[0].split('-',2)
    else:
        mods = np.array([argv[0],argv[0]])
    if mods is None:
        print('Incorrect input models...exiting...')
        sys.exit()
    else:
        print('Fitting ' + modstr[int(mods[0])-1] + ' model to ' + modstr[int(mods[1])-1] + ' data') 
        res = fitPar(int(mods[0]),int(mods[1]))
        return(res)


if __name__ == "__main__":
   main(sys.argv[1:])