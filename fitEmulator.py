from mission import mission
import numpy as np
import sys

def main(argv):
    mis = mission('nidelva_27_05_21')
    mis.emulator(model=int(argv[0]),pars = True)
    mis.mean(version2 = "tfc")
    if int(argv[0])==4:
        par = np.hstack([[-1.1]*27,[-1.2]*27,[5.1]*27,[4.1]*27,[0.04]*27,[0.4]*27,[0.6]*27,5])
    elif int(argv[0]) == 2:
        par = np.array([-1,-0.8,5,4,0.04,0.4,0.6,4])
    mis.fit(par= par, verbose = True)
    print("finished")

if __name__ == "__main__":
   main(sys.argv[1:])