
import sys, getopt
import numpy as np
from spde import spde
from joblib import Parallel, delayed
from functools import partial
import os


def fit(version,mod,data,vers):
    try: 
        res = mod.fitTo(data,vers[version,1],vers[version,2], vers[version,0],verbose = False)
    except:
        print("Model "+ str(mod.model) + " data " + str(data) + " dho " + str(vers[version,1])+ " r " + str(vers[version,2]) + " num " + str(vers[version,0]) + " crashed")
        return(False)
    else:
        return(res)

def fitPar(model,data):
    vers = findFits(model,data)
    mod = spde(model = model)
    fit_ = partial(fit, mod=mod, data = data, vers=vers)
    res = Parallel(n_jobs=40,verbose = 100)(delayed(fit_)(i) for i in range(vers.shape[0]))
    return(res)

def findFits(model, data):
    vers = np.array([[i,j,k] for i in range(1,101) for j in range(1,4) for k in range(1,4)])
    modstr = np.array(["SI", "SA", "NA1"])
    dho = np.array(["100","10000","27000"])
    r = np.array(["1","10","100"])
    print(vers.shape)
    for file in os.listdir("./fits/"):
        if file.startswith(modstr[model-1]+"-"+modstr[data-1]):
            tmp = file.split('-')[2:]
            tdho = np.where(dho == tmp[0][3:])[0][0] + 1
            tnum = int(tmp[2].split(".")[0])
            tr = np.where(r==tmp[1][1:])[0][0]+1
            tar = np.array([tnum,tdho,tr]) # num , dho , r
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