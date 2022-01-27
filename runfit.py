
import sys, getopt
import numpy as np
from spde import spde
from joblib import Parallel, delayed
from functools import partial
from datetime import datetime
from tqdm import tqdm


def fit(version,mod,data,vers):
    try: 
        res = mod.fitTo(data,vers[version,1],vers[version,2], vers[version,0],verbose = False)
    except:
        print("Model "+ mod + " data " + data + " dho " + vers[version,1]+ " r " + vers[version,2] + " num " + vers[version,0] + " crashed")
        return(False)
    else:
        return(res)

def fitPar(model,data,lower, upper):
    vers = np.array([[i,j,k] for i in range(lower,upper) for j in range(1,4) for k in range(1,4)])
    mod = spde(model = model)
    fit_ = partial(fit, mod=mod, data = data, vers=vers)
    res = Parallel(n_jobs=20)(delayed(fit_)(i) for i in range(vers.shape[0]))
    return(res)

def main(argv):
    modstr = ["Stationary Isotropic", "Stationary Anistropic", "Non-stationary Simple Anisotropic","Non-stationary Complex Anisotropic"]
    mods = None
    bonds = np.array(['0','100'])
    if "-" in argv[0]:
        mods = argv[0].split('-',2)
    else:
        mods = np.array([argv[0],argv[0]])
    if len(argv)==2:
        if "-" in argv[1]:
            bound = argv[1].split('-',2)
        else:
            print("Incorrect upper and lowers bounds...exiting...")
            sys.exit()
    if mods is None:
        print('Incorrect input models...exiting...')
        sys.exit()
    else:
        print('Fitting ' + modstr[int(mods[0])-1] + ' model to ' + modstr[int(mods[1])-1] + ' data') 
        res = fitPar(int(mods[0]),int(mods[1]),int(bonds[0]),int(bonds[1]))
        return(res)


if __name__ == "__main__":
   main(sys.argv[1:])