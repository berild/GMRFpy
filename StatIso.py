import sys, getopt
import numpy as np
from scipy import sparse
from ah3d2 import AH
from sksparse.cholmod import cholesky
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
inla = importr("INLA")
#robj.r('inla.setOption("smtp" = "pardiso", pardiso.license = "~/OneDrive - NTNU/host_2020/pardiso.lic")')
from scipy.optimize import minimize
import os
from grid import Grid

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
    tmp = Q.shape
    Q = Q.tocoo()
    r = Q.row
    c = Q.col
    v = Q.data
    tmpQinv = np.array(robj.r["as.data.frame"](robj.r["summary"](robj.r["inla.qinv"](robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))))
    return(sparse.csc_matrix((np.array(tmpQinv[2,:],dtype = "float32"), (np.array(tmpQinv[0,:]-1,dtype="int32"), np.array(tmpQinv[1,:]-1,dtype="int32"))), shape=tmp))

class StatIso:
    #mod1: kappa(0), gamma(1), sigma(2)
    def __init__(self,grid=None,par=None):
        assert par is None or (par.size==3 if par is not None else False)
        self.grid = Grid() if grid is None else grid
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.kappa = np.log(0.5) if par is None else par[0]
        self.gamma = np.log(0.5) if par is None else par[1]
        self.sigma = np.log(0.5) if par is None else par[2]
        self.tau = np.log(0.5) if par is None else par[2]
        self.Dv = self.V*sparse.eye(self.n)
        self.iDv = 1/self.V*sparse.eye(self.n)
        self.Q = None
        self.Q_fac = None
        self.mvar = None
        self.data = None
        self.r = None
        self.S = None
        self.opt_steps = 0
        self.verbose = False
        self.grad = True
        self.like = 10000
        self.jac = np.array([-100]*3)
    
    def load(self,simple = False):
        simmod = np.load("./simmodels/SI.npz")
        self.kappa = simmod['kappa']*1
        self.gamma = simmod['gamma']*1
        self.sigma = simmod['sigma']*1
        self.tau = np.log(1/np.exp(self.sigma)**2)
        if not simple:
            Hs = np.exp(self.gamma)*np.eye(3) + np.zeros((self.n,6,3,3))
            Dk =  sparse.diags([np.exp(self.kappa)]*self.n) 
            A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
            Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
            A_mat = self.Dv@Dk - Ah
            self.Q = A_mat.transpose()@self.iDv@A_mat
            self.Q_fac = cholesky(self.Q)
            self.mvar = rqinv(self.Q).diagonal()
    
    # maybe add some assertions
    def loadFit(self, simmod, dho, r, num, file = None):
        if file is None:
            mods = np.array(['SI','SA','NA1','NA2'])
            dhos = np.array(['100','10000','27000'])
            rs = np.array([1,10,100])
            file = './fits/' + mods[simmod-1] + '-SI-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = fitmod['S']*1
        par =fitmod['par']*1
        self.kappa = par[0]
        self.gamma = par[1]
        self.tau = par[2]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = np.exp(par[1])*np.eye(3) + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags([np.exp(par[0])]*self.n) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n,self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)
        self.mvar = rqinv(self.Q).diagonal()

    def fitTo(self,simmod,dho,r,num,verbose = False, grad = True, par = None):
        if par is None:
            par = np.array([-1,0.5,2])
        mods = np.array(['SI','SA','NA1','NA2'])
        dhos = np.array(['100','10000','27000'])
        rs = np.array([1,10,100])
        tmp = np.load('./simulations/' + mods[simmod-1] + '-'+str(num)+".npz")
        self.data = (tmp['data']*1)[np.sort(tmp['locs'+dhos[dho-1]]*1),:(rs[r-1])]
        self.r = rs[r-1]
        self.S = np.zeros((self.n))
        self.S[np.sort(tmp['locs'+dhos[dho-1]]*1)] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        res = self.fit(data = self.data, r=self.r, S = self.S,verbose = verbose, grad = grad,par = par)
        np.savez(file = './fits/' + mods[simmod-1] + '-SI-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res['x'], S = self.S)
        return(True)

    # implement S either locations, the array or not specified but data must be nans
    def fit(self,data, r, S = None, par = None,verbose = False, grad = True):
        if par is None:
            par = np.array([-1,0.5,2])
        assert S is not None
        self.data = data
        self.r = r  
        self.S = S
        self.opt_steps = 0
        self.grad = grad
        self.verbose = verbose
        if self.grad:
            res = minimize(self.logLike, x0 = par,jac = True, method = "BFGS",tol = 1e-3)
        else:    
            res = minimize(self.logLike, x0 = par, tol = 1e-3)
        self.kappa = res['x'][0]
        self.gamma = res['x'][1]
        self.tau = res['x'][2]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def getPars(self):
        return(np.hstack([self.kappa,self.gamma,self.tau]))

    def logLike(self, par):
        data  = self.data
        Hs = np.exp(par[1])*np.eye(3) + np.zeros((self.n,6,3,3))
        Dk =  np.exp(par[0])*sparse.eye(self.n) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        Q = A_mat.transpose()@self.iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[2])
        Q_fac = self.cholesky(Q)
        Q_c_fac= self.cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if self.grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[2]))
        if self.r == 1:
            data = data.reshape(data.shape[0],1)
            mu_c = mu_c.reshape(mu_c.shape[0],1)
        if self.grad:
            Qinv = rqinv(Q)
            Qcinv = rqinv(Q_c)

            A_kappa = Dk@self.Dv
            A_gamma = - Ah

            Q_kappa = A_kappa.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_kappa
            Q_gamma = A_gamma.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_gamma

            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[2]/2 - 1/2*Q_c_fac.logdet()*self.r
            g_kappa = 1/2*((Qinv - Qcinv)@Q_kappa).diagonal().sum()*self.r
            g_gamma = 1/2*((Qinv - Qcinv)@Q_gamma).diagonal().sum()*self.r
            g_noise = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[2])).diagonal().sum()*self.r

            for j in range(self.r): # Maybe make a better version than this for loop possibly need to account for dimension 0
                g_kappa = g_kappa + (- 1/2*mu_c[:,j].transpose()@Q_kappa@mu_c[:,j])
                g_gamma = g_gamma + (- 1/2*mu_c[:,j].transpose()@Q_gamma@mu_c[:,j])
                g_noise = g_noise + (- 1/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j] - self.S@mu_c[:,j])*np.exp(par[2]))
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[2])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = -like/(self.S.shape[0]*self.r)
            jac = - np.array([g_kappa,g_gamma,g_noise])/(self.S.shape[0]*self.r)
            #jac = jac/np.linalg.norm(jac)
            self.opt_steps = self.opt_steps + 1
            self.like = like
            self.jac = jac
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3 = %2.2f"%np.exp(par[1]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[2])))
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[2]/2 - 1/2*Q_c_fac.logdet()*self.r
            for j in range(self.r):
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[2])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = -like/(self.S.shape[0]*self.r)
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3 = %2.2f"%np.exp(par[1]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[2])))
            return(like)

    def setQ(self,par = None):
        if par is None:
            assert(self.kappa is not None and self.gamma is not None and self.sigma is not None)
        else:
            self.kappa = par[0]
            self.gamma = par[1]
            self.sigma = par[2]
        Hs = np.exp(self.gamma)*np.eye(3) + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags([np.exp(self.kappa)]*self.n) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)
        

    def sim(self):
        self.load()
        mods = []
        for file in os.listdir("./simulations/"):
            if file.startswith("SI-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/SI-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False), locs27000 = np.arange(self.n))
        return(True)
        

    def sample(self,n = 1, par = None):
        if par is None:
            assert(self.kappa is not None and self.gamma is not None and self.sigma is not None)
        if self.Q is None or self.Q_fac is None:
            self.setQ(par = par)
        data = np.zeros((self.n,n))
        for i in range(n):
            z = np.random.normal(size = self.n)
            data[:,i] = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n)*np.exp(self.sigma)
        return(data)
    
    def cholesky(self,Q):
        try: 
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)
            