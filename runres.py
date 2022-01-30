import sys, getopt
import numpy as np
import os

def print1(res):
    lines = list(["\u03BA       ","\u03B3       ","\u03C4       "])
    for j in range(3):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "|  %.2f"%res[0][j,i] + "(%.2f) "%res[1][j,i] 

    print("DHO     |                    100                  |                   10000                 |                   27000          \n")
    print("Real.   |      1      |      10     |      100    |      1      |      10     |     100     |      1      |      10     |      100  \n")
    print(lines[0])
    print(lines[1])
    print(lines[2])

def main(argv):
    vers = np.array([[i,j,k] for i in range(1,101) for j in range(1,4) for k in range(1,4)])
    modstr = np.array(["SI", "SA", "NA"])
    dho = np.array(["100","10000","27000"])
    r = np.array(["1","10","100"])
    model = int(argv[0])
    count = 0
    npars = 0
    for file in os.listdir("./fits/"):
        if file.startswith(modstr[model-1]+"-"+modstr[model-1]):
            count = count + 1
            if count == 1:
                npars = (np.load("./fits/"+file)['par']*1).shape[0]
    pars = np.zeros((count,npars + 2))
    count = 0
    for file in os.listdir("./fits/"):
        if file.startswith(modstr[model-1]+"-"+modstr[model-1]):
            par = (np.load("./fits/"+file)['par']*1)
            tmp = file.split('-')[2:]
            tdho = np.where(dho == tmp[0][3:])[0][0] + 1
            tr = np.where(r==tmp[1][1:])[0][0] + 1
            pars[count,:] = np.hstack([par,tdho,tr])
            count = count + 1
    np.savez(file = modstr[model-1]+"-"+modstr[model-1] + "-pars",pars = pars)
    res = list([np.zeros((npars,9)),np.zeros((npars,9))])
    for i in range(npars): #dho
        for j in range(npars): #r
            res[0][:,(i)*npars + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars].mean(axis=1)
            res[1][:,(i)*npars + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars].std(axis=1)
    np.savez(file = modstr[model-1]+"-"+modstr[model-1] + "-results",res = res)
    return(True)

if __name__ == "__main__":
   main(sys.argv[1:])