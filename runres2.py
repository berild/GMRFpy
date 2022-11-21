import sys
import numpy as np
import os
from spde import spde
from scipy import sparse
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
importr("Matrix")
robj.r.source("rqinv.R")
from scipy.stats import norm

def rqinv(Q):
    tshape = Q.shape
    Q = Q.tocoo()
    r = Q.row
    c = Q.col
    v = Q.data
    tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
    return(sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape))


def main():
    modstr = np.array(["SI", "SA", "NA"])
    dho = np.array(["100","10000"])
    r = np.array(["1","10"])
    for i in range(len(modstr)):
        for j in range(len(dho)):
            for k in range(len(r)):
                for l in np.arange(1,101):
                    if os.path.exists("./fits/s2-"+modstr[i] +'-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npy"):
                        continue
                    else:
                        mod1 = os.path.exists("./fits/NA-" + modstr[i] + '-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npz")
                        mod2 = os.path.exists("./fits/SA-" + modstr[i] + '-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npz")
                        mod3 = os.path.exists("./fits/SI-" + modstr[i] + '-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npz")
                        if mod1 and mod2 and mod3:
                            print("Running " + modstr[i]+ " " + str(l))
                            res = np.zeros(3)
                            for infmod in [1,2,4]:
                                mod = spde(model = infmod)
                                tmp = np.load('./simulations/' + modstr[i] + '-'+str(l)+".npz")
                                data = (tmp['data']*1)[tmp['locs'+dho[j]],:(int(r[k]))]
                                ks = tmp['locs'+dho[j]]
                                test = (tmp['data']*1)[:,:(int(r[k]))]
                                test = np.delete(test,tmp['locs'+dho[j]],axis = 0)
                                mod.loadFit(file = "./fits/"+ modstr[2 if infmod == 4 else infmod-1]+ "-" + modstr[i] + '-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npz")
                                mu = np.zeros(mod.mod.n)
                                Q = mod.mod.Q.copy()
                                Q_fac = mod.mod.Q_fac
                                S = sparse.eye(mod.mod.n).tocsc()[ks,:]
                                Q = Q + S.transpose()@S*1/np.exp(mod.mod.sigma)**2
                                Q_fac.cholesky_inplace(Q)
                                pred = - Q_fac.solve_A(S.transpose().tocsc())@((S@mu)[:,np.newaxis] - data)*1/np.exp(mod.mod.sigma)**2
                                pred = np.delete(pred,tmp['locs'+dho[j]],axis = 0)
                                res[2 if infmod == 4 else infmod-1] = np.sqrt(np.mean((pred-test)**2))
                                #sigma = np.delete(np.sqrt(rqinv(Q).diagonal()),tmp['locs'+dho[j]])
                                #z = (test - pred)/sigma[:,np.newaxis]
                                #res[2 if infmod == 4 else infmod-1,1] = np.mean(sigma[:,np.newaxis]*(- 2/np.sqrt(np.pi) + 2*norm.pdf(z) + z*(2*norm.cdf(z)-1)))
                        np.save("./fits/s2-"+modstr[i] +'-dho'+dho[j] +'-r' + r[k]+"-"+ str(l)+".npy",res)
                        print("Finished " + modstr[i]+ " " +str(l))
    return(True)

if __name__ == "__main__":
   main()