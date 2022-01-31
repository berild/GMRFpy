
import sys, getopt
import numpy as np
import multiprocessing as mp
from spde import spde



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
            res = np.zeros(100)
            for i in range(100):
                res[i] = mod.sim()
            return((res.sum()==100))
    else:
        print("Incorrect input arguments...")
        sys.exit()


if __name__ == "__main__":
   main(sys.argv[1:])