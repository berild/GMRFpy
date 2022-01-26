
import sys, getopt
import numpy as np
from spde import spde
from joblib import Parallel, delayed
from functools import partial


def fit(version,mod,data,vers):
    try: 
        res = mod.fitTo(data,vers[version,1],vers[version,2], vers[version,0],verbose = False)
    except:
        print("Model "+ mod + " data " + data + " dho " + vers[version,1]+ " r " + vers[version,2] + " num " + vers[version,0] + " crashed")
        return(False)
    else:
        return(res)

def fitSelf(model):
    vers = np.array([[i,j,k] for i in range(1,101) for j in range(1,4) for k in range(1,4)])
    mod = spde(model = model)
    fit_ = partial(fit, mod=mod, data = model, vers=vers)
    res = Parallel(n_jobs=20)(delayed(fit_)(i) for i in range(vers.shape[0]))
    return(res)
    

def fitOther(model,data):
    vers = np.array([[i,j,k] for i in range(1,101) for j in range(1,4) for k in range(1,4)])
    mod = spde(model = model)
    fit_ = partial(fit, mod=mod, data = data, vers=vers)
    res = Parallel(n_jobs=20)(delayed(fit_)(i) for i in range(vers.shape[0]))
    return(res)

def main(argv):
    modstr = ["Stationary Isotropic", "Stationary Anistropic", "Non-stationary Simple Anisotropic","Non-stationary Complex Anisotropic"]
    if (len(argv)==0):
        print("No simulation model specified...exiting...")
        sys.exit(2)
    elif (len(argv)==1):
        if ((int(argv[0])<1) or (int(argv[0])>4)):
            print("Incorrect simulation model...exiting...")
            sys.exit()
        else:
            print("Fitting the " + modstr[int(argv[0])-1] + " model to own data...")
            res = fitSelf(int(argv[0]))
            print(sum(res)==900)
            return(sum(res)==900)
    elif (len(argv)==2):
        if ((int(argv[0])<1) or (int(argv[0])>4)):
            print("Incorrect simulation model...exiting...")
            sys.exit()
        else:
            print("Fitting the " + modstr[int(argv[0])-1] + " model to " + modstr[int(argv[1])-1] + " data")
            res = fitOther(int(argv[0]),int(argv[1]))
            print(sum(res)==900)
            return(sum(res)==900)
    else:
        print("To many input arguments...")
        sys.exit()


if __name__ == "__main__":
   main(sys.argv[1:])