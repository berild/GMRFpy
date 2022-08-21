import sys, getopt
import numpy as np
import os
from spde import spde

def print1(res):
    lines = list(["\u03BA      ","\u03B3      ","\u03C4      "])
    for j in range(3):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %5.4f"%res[0][j,i] + "(%.5f) "%res[1][j,i]

    print("DHO    |                       100                      |                     10000                      |                     27000             \n")
    print("Real.  |       1       |       10       |      100      |       1       |       10       |      100      |       1       |       10       |       100    \n")
    print(lines[0])
    print(lines[1])
    print(lines[2])

def print2(res):
    lines = list(["\u03BA      ","\u03B3      ","vx      ","vy      ","vz     ","\u03C1_1      ","\u03C1_2      ","\u03C4      "])
    for j in range(8):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %5.4f"%res[0][j,i] + "(%.5f) "%res[1][j,i]

    print("DHO    |                       100                      |                     10000                      |                     27000             \n")
    print("Real.  |       1       |       10       |      100      |       1       |       10       |      100      |       1       |       10       |       100    \n")
    for j in range(8):
        print(lines[j])

def print3(res):
    lines = list(["\u03BA      ","\u03B3_1      ","\u03B3_2      ","\u03B3_3      ","\u03C1_1      ","\u03C1_2      ","\u03C1_3      ","\u03C4      "])
    for j in range(8):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %5.4f"%res[0][j,i] + "(%.5f) "%res[1][j,i]

    print("DHO    |                       100                      |                     10000                      |                     27000             \n")
    print("Real.  |       1       |       10       |      100      |       1       |       10       |      100      |       1       |       10       |       100    \n")
    for j in range(8):
        print(lines[j])
        

def print4(res):
    lines = list(["\u03BA      ","\u03B3      ","vx      ","vy      ","vz     ","\u03C1_1      ","\u03C1_2      ","\u03C4      "])
    for j in range(8):
        lines.append("")
        for i in range(9):
            lines[j] = lines[j] + "| %5.4f"%res[0][j,i] + "(%.5f) "%res[1][j,i]

    print("DHO    |                       100                      |                     10000                      |                     27000             \n")
    print("Real.  |       1       |       10       |      100      |       1       |       10       |      100      |       1       |       10       |       100    \n")
    for j in range(8):
        print(lines[j])


def main(argv):
    vers = np.array([[i,j,k] for i in range(1,101) for j in range(1,4) for k in range(1,4)])
    modstr = np.array(["SI", "SA", "NI", "NA"])
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
    mod = spde(model = model)
    mod.load(simple = True)
    truth = mod.getPars()
    for i in range(3): #dho
        for j in range(3): #r
            res[0][:,(i)*3 + j] = pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars].mean(axis=1) - truth
            res[1][:,(i)*3 + j] = np.mean((pars[np.where(((pars[:,npars]==(i+1))&(pars[:,npars+1]==(j+1)))),:npars] - truth)**2,axis = 1)
    np.savez(file = modstr[model-1]+"-"+modstr[model-1] + "-results",res = res)
    if model == 1:
        print1(res)
    elif model == 2:
        print2(res)
    elif model == 3:
        print3(res)
    elif model == 4:
        print4(res)
    return(True)

if __name__ == "__main__":
   main(sys.argv[1:])