import sys
import numpy as np
from spde import spde
from tqdm import tqdm



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
            mod.load()
            res = np.zeros(100)
            for i in tqdm(range(100)):
                res[i] = mod.sim(verbose = False)
    else:
        print("Incorrect input arguments...")
        sys.exit()


if __name__ == "__main__":
   main(sys.argv[1:])