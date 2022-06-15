#from re import S
import numpy as np
from scipy import sparse
from ah3d2 import AH
from sksparse.cholmod import cholesky
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
importr("Matrix")
#robj.r('inla.setOption("smtp" = "pardiso", pardiso.license = "~/OneDrive - NTNU/host_2020/pardiso.lic")')
#import nlopt
import os
from grid import Grid
robj.r.source("rqinv.R")
from scipy.optimize import minimize

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = (indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def rqinv(Q):
    tshape = Q.shape
    Q = Q.tocoo()
    r = Q.row
    c = Q.col
    v = Q.data
    tmpQinv =  np.array(robj.r.rqinv(robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))
    return(sparse.csc_matrix((np.array(tmpQinv[:,2],dtype = "float32"), (np.array(tmpQinv[:,0],dtype="int32"), np.array(tmpQinv[:,1],dtype="int32"))), shape=tshape))


class NonStatIso:
    #mod3: kappa(0:27), gammaXY (27),gammaZ (28), sigma(29)
    def __init__(self,grid=None,par=None):
        assert par is None or (par.size==30 if par is not None else False)
        self.grid = Grid() if grid is None else grid
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.kappa = np.array([0.5]*27) if par is None else par[0:27]
        self.gammaXY = 0.5 if par is None else par[27]
        self.gammaZ = 0.5 if par is None else par[28]
        self.tau = 0.5 if par is None else par[29]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.Dv = self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.iDv = 1/self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.grid.basisH()
        self.grid.basisN()
        self.Q = None
        self.Q_fac = None
        self.mvar = None
        self.data = None
        self.r = None
        self.S = None
        self.opt_steps = 0
        self.verbose = False
        self.grad = True
        self.like = -10000
        self.jac = np.array([-100]*31)
        self.loaded = False

    def setGrid(self,grid):
        self.grid = grid
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.Dv = self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.iDv = 1/self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.grid.basisH()
        self.grid.basisN()

    
    def getPars(self):
        return(np.hstack([self.kappa,self.gammaXY,self.gammaZ,self.tau]))

    def load(self,simple = False):
        if self.loaded:
            return
        simmod = np.load("./simmodels/NI.npz")
        self.kappa = simmod['kappa']*1
        self.gammaXY = simmod['gammaXY']*1
        self.gammaZ = simmod['gammaZ']*1
        self.sigma = simmod['sigma']*1
        self.tau = np.log(1/np.exp(self.sigma)**2)
        if not simple:
            Hs = self.getH()
            Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa)))
            A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
            Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
            A_mat = self.Dv@Dk - Ah
            self.Q = A_mat.transpose()@self.iDv@A_mat
            self.Q_fac = self.cholesky(self.Q)
            assert(self.Q_fac != -1)
            self.mvar = rqinv(self.Q).diagonal()
            self.loaded = True

    def fit(self,data, r, S = None,verbose = False, fgrad = True, par = None):
        #mod3: kappa(0:27), gammaXY (27),  gammaZ (28), sigma(29)
        if par is None:
            par = np.hstack([[0.5]*27,2,-1,4])
        self.data = data
        self.r = r
        self.S = S
        self.opt_steps = 0
        self.grad = fgrad
        self.verbose = verbose
        if self.grad:
            res = minimize(self.logLike, x0 = par,jac = True, method = "BFGS",tol = 1e-4)
            res = res['x']
        else:    
            res = minimize(self.logLike, x0 = par)#, tol = 1e-3)
            res = res['x']
        self.kappa = res[0:27]
        self.gammaXY = res[27]
        self.gammaZ = res[28]
        self.tau = res[29]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def fitTo(self,simmod,dho,r,num,verbose = False, fgrad = True, par = None):
        if par is None:
            par = np.array([-0.5]*30)
        mods = np.array(['SI','SA','NI','NA'])
        dhos = np.array(['100','10000','27000'])
        rs = np.array([1,10,100])
        tmp = np.load('./simulations/' + mods[simmod-1] + '-'+str(num)+".npz")
        self.data = (tmp['data']*1)[tmp['locs'+dhos[dho-1]],:(rs[r-1])]
        self.r = rs[r-1]
        self.S = np.zeros((self.n))
        self.S[tmp['locs'+dhos[dho-1]]*1] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        res = self.fit(data = self.data, r=self.r, S = self.S,verbose = verbose, fgrad = fgrad,par = par)
        np.savez('./fits/' + mods[simmod-1] + '-NI-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res)
        return(True)

    # assertion for number of parameters
    def loadFit(self, simmod, dho, r, num, file = None):
        if file is None:
            mods = np.array(['SI','SA','NI','NA'])
            dhos = np.array(['100','1000','10000'])
            rs = np.array([1,10,100])
            file = './fits/' + mods[simmod-1] + '-NI-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = np.zeros((self.grid.M*self.grid.N*self.grid.P))
        self.S[fitmod['S']*1] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        par =fitmod['par']*1
        self.kappa = par[0:27]
        self.gammaXY = par[27]
        self.gammaZ = par[28]
        self.tau = par[29]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = self.getH()
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n,self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)
        self.mvar = rqinv(self.Q).diagonal()

    def sample(self,n = 1, par = None):
        if par is None:
            assert(self.kappa is not None and self.gammaXY is not None and self.gammaZ is not None and self.sigma is not None)
        if self.Q is None or self.Q_fac is None:
            self.setQ(par = par)
        data = np.zeros((self.n,n))
        for i in range(n):
            z = np.random.normal(size = self.n)
            data[:,i] = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n)*np.exp(self.sigma)
        return(data)

    def sim(self):
        self.load()
        mods = []
        for file in os.listdir("./simulations/"):
            if file.startswith("NI-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/NI-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False), locs27000 = np.arange(self.n))
        return(True)

    def setQ(self,par = None):
        if par is None:
            assert(self.kappa is not None and self.gammaXY is not None and self.gammaZ is not None and self.sigma is not None)
        else:
            self.kappa = par[0:27]
            self.gammaXY = par[27]
            self.gammaZ = par[28]
            self.tau = par[29]
            self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = np.diag([np.exp(self.gammaXY),np.exp(self.gammaXY),np.exp(self.gammaZ)]) + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)

    def cholesky(self,Q):
        try: 
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)

    def logLike(self, par):
        #mod3: kappa(0:27), gammaXY (27), gammaZ (28), sigma(29)
        data  = self.data
        Hs = np.diag([np.exp(par[27]),np.exp(par[27]),np.exp(par[28])]) + np.zeros((self.n,6,3,3))
        lkappa = self.grid.evalB(par = par[0:27])
        Dk =  sparse.diags(np.exp(lkappa)) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        del A_H
        A_mat = self.Dv@Dk - Ah
        Q = A_mat.transpose()@self.iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[29])
        Q_fac = self.cholesky(Q)
        Q_c_fac= self.cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if self.grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[29]))
        if self.r == 1:
            data = data.reshape(data.shape[0],1)
            mu_c = mu_c.reshape(mu_c.shape[0],1)
        if self.grad:
            Qinv = rqinv(Q)
            Qcinv = rqinv(Q_c)

            Hs_par = np.diag([np.exp(par[27]),np.exp(par[27]),0]) + np.zeros((self.n,6,3,3))
            A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            A_par = - sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
            del A_H_par
            Q_gammaXY = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par
            Hs_par = np.diag([0,0,np.exp(par[28])]) + np.zeros((self.n,6,3,3))
            A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
            A_par = - sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
            del A_H_par
            Q_gammaZ = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par

            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[29]/2 - 1/2*Q_c_fac.logdet()*self.r
            g_par = np.zeros(30)
            g_par[27] = 1/2*((Qinv - Qcinv)@Q_gammaXY).diagonal().sum()*self.r
            g_par[28] = 1/2*((Qinv - Qcinv)@Q_gammaZ).diagonal().sum()*self.r
            g_par[29] = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[29])).diagonal().sum()*self.r
            for i in range(27):
                Dk2 = sparse.diags(self.grid.bs[:,i]*np.exp(lkappa))
                A_par = self.Dv@Dk2
                Q_par = A_par.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_par
                g_par[i] = 1/2*((Qinv - Qcinv)@Q_par).diagonal().sum()*self.r
                for j in range(self.r): 
                    g_par[i] = g_par[i] + (- 1/2*mu_c[:,j].transpose()@Q_par@mu_c[:,j])
                    if i==0:
                        # gammaX Y Z and tau
                        g_par[27] = g_par[27] + (- 1/2*mu_c[:,j].transpose()@Q_gammaXY@mu_c[:,j])
                        g_par[28] = g_par[28] + (- 1/2*mu_c[:,j].transpose()@Q_gammaZ@mu_c[:,j])
                        g_par[29] = g_par[29] + (- 1/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j] - self.S@mu_c[:,j])*np.exp(par[29]))
                        like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[29])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like =  -like/(self.S.shape[0]*self.r)
            jac =  -g_par/(self.S.shape[0]*self.r)
            self.opt_steps = self.opt_steps + 1
            del Q_fac
            del Q_c_fac
            del Qinv
            del Qcinv
            np.savez('SINMOD-NI-new.npz', par = par)
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%(np.mean(np.exp(lkappa))), "\u03B3_XY = %2.2f"%np.exp(par[27]), "\u03B3_Z = %2.2f"%np.exp(par[28]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[29])))
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[29]/2 - 1/2*Q_c_fac.logdet()*self.r
            for j in range(self.r):
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[29])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = like/(self.S.shape[0]*self.r)
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like))#, "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3 = %2.2f"%np.exp(par[1]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[2])))
            return(like)