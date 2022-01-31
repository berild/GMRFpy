
import sys, getopt
import numpy as np
import multiprocessing as mp
from spde import spde
from joblib import Parallel, delayed


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
            print("Simulating from " + modstr[int(argv[0])-1] + "...")
            mod = spde(model = int(argv[0]))
            res = Parallel(n_jobs=30,verbose = 100)(delayed(mod.fit)() for i in range(100))
            return(res)
    else:
        print("Incorrect input arguments...")
        sys.exit()


if __name__ == "__main__":
   main(sys.argv[1:])