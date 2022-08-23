from mission import mission
import numpy as np

def main():
    mis = mission('nidelva_27_05_21')
    mis.emulator(model=4,pars = True)
    mis.mean(version2 = "tfc")
    par = np.hstack([[-1.1]*27,[-1.2]*27,[5.1]*27,[4.1]*27,[0.04]*27,[0.4]*27,[0.6]*27,5])
    mis.fit(par= par, verbose = True)
    print("finished")

if __name__ == "__main__":
   main()