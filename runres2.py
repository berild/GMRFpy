import sys
import numpy as np
import os
from spde import spde
from scipy import sparse


def main(argv):
    modstr = np.array(["SI", "SA", "NA"])
    dho = np.array(["100","10000","27000"])
    rs = np.array(["1","10","100"])
    if not os.path.exists("completed2.npy"):
        completed = np.zeros(9,dtype = "bool")
    else: 
        completed = np.load("completed2.npy")
    if not os.path.exists("results2.npy"):
        res = np.zeros((2,9,4))
    else:
        res = np.load('results2.npy')
    for simmod in np.arange(1,4):
        for infmod in np.arange(1,4):
            if completed[(simmod-1)*3 + infmod-1]:
                continue
            print("Simmod: " + modstr[simmod-1] + " | infmod: " + modstr[infmod-1])
            for tdho in np.arange(1,3):
                for tr in np.arange(1,3):
                    print("DHO: " + dho[tdho-1] + " | r: " + rs[tr-1])
                    tres = []
                    mres = []
                    count = 0
                    for file in os.listdir("./fits/"):
                        if file.startswith(modstr[infmod-1]+"-"+modstr[simmod-1]+'-dho'+dho[tdho-1]+'-r' + rs[tr-1]):
                            if os.path.exists("./preds/"+ modstr[infmod-1]+"-"+modstr[simmod-1]+'-dho'+dho[tdho-1]+'-r' + rs[tr-1] + '.npz'):
                                tmpres = np.load("./preds/"+ modstr[infmod-1]+"-"+modstr[simmod-1]+'-dho'+dho[tdho-1]+'-r' + rs[tr-1] + '.npz')
                                tres.append(tmpres['pred'])
                                mres.append(tmpres['mvar'])
                            else:   
                                mod = spde(model = infmod)
                                tmp = file.split('-')[2:]
                                tnum = int(tmp[2][:(len(tmp[2])-4)])
                                tmp = np.load('./simulations/' + modstr[simmod-1] + '-'+str(tnum)+".npz")
                                data = (tmp['data']*1)[tmp['locs'+dho[tdho-1]],:(int(rs[tr-1]))]
                                ks = tmp['locs'+dho[tdho-1]]
                                test = (tmp['data']*1)[:,:(int(rs[tr-1]))]
                                test = np.delete(test,tmp['locs'+dho[tdho-1]],axis = 0)

                                mod.loadFit(file = "./fits/" + file)
                                mu = np.zeros(mod.mod.n)
                                Q = mod.mod.Q.copy()
                                Q_fac = mod.mod.Q_fac
                                S = sparse.eye(mod.mod.n).tocsc()[ks,:]
                                Q = Q + S.transpose()@S*1/np.exp(mod.mod.sigma)**2
                                Q_fac.cholesky_inplace(Q)
                                pred = - Q_fac.solve_A(S.transpose().tocsc())@((S@mu)[:,np.newaxis] - data)*1/np.exp(mod.mod.sigma)**2
                                pred = np.delete(pred,tmp['locs'+dho[tdho-1]],axis = 0)
                                tres.append(np.mean((pred-test)**2))
                                mod.Mvar()
                                mres.append(np.std(mod.mod.mvar))
                                if os.path.exists("./preds"):
                                    os.mkdir("./preds")
                                np.savez("./preds/"+ modstr[infmod-1]+"-"+modstr[simmod-1]+'-dho'+dho[tdho-1]+'-r' + rs[tr-1] + '.npz',pred = np.mean((pred-test)**2), mvar = np.std(mod.mod.mvar))
                    res[0,(simmod-1)*3 + infmod - 1, (tdho-1)*2 + tr - 1] = np.mean(tres)
                    res[1,(simmod-1)*3 + infmod - 1, (tdho-1)*2 + tr - 1] = np.mean(mres)
                    np.save('results2.npy',res)
                    if count == 400:
                        completed[(simmod-1)*3 + infmod - 1] = True
                        np.save('completed2.npy',completed)
    return(True)

if __name__ == "__main__":
   main()