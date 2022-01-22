import sys, getopt
import numpy as np
from scipy import sparse
from ah3d2 import AH
from sksparse.cholmod import cholesky
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
inla = importr("INLA")
robj.r('inla.setOption("smtp" = "pardiso", pardiso.license = "~/OneDrive - NTNU/host_2020/pardiso.lic")')
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


class NonStatAnIsoComp:
    #mod4: kappa(0:27), gamma1(27:54), gamma2(54:81), gamma3(81:108), rho1(108:135), rho2(135:162), rho3(162:189), sigma(189)
    def __init__(self,grid=None,par=None):
        assert par is None or (par.size==190 if par is not None else False)
        self.grid = Grid() if grid is None else grid
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.kappa = np.array([0.5]*27) if par is None else par[0:27]
        self.gamma1 = np.array([0.5]*27) if par is None else par[27:54]
        self.gamma2 = np.array([0.5]*27) if par is None else par[54:81]
        self.gamma3 = np.array([0.5]*27) if par is None else par[81:108]
        self.rho1 = np.array([0.5]*27) if par is None else par[108:135]
        self.rho2 = np.array([0.5]*27) if par is None else par[135:162]
        self.rho3 = np.array([0.5]*27) if par is None else par[162:189]
        self.tau = 0.5 if par is None else par[189]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.Dv = self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.iDv = 1/self.V*sparse.eye(self.grid.M*self.grid.N*self.grid.P)
        self.parH = self.grid.basisH()
        self.parKappa = self.grid.basisN()
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
        self.jac = np.array([-100]*190)

    def load(self):
        simmod = np.load("./simmodels/NA2.npz")
        self.kappa = simmod['kappa']*1
        self.gamma1 = simmod['gamma1']*1
        self.gamma2 = simmod['gamma2']*1
        self.gamma3 = simmod['gamma3']*1
        self.rho1 = simmod['rho1']*1
        self.rho2 = simmod['rho2']*1
        self.rho3 = simmod['rho3']*1
        self.sigma = simmod['sigma']*1
        self.tau = np.log(1/np.exp(self.sigma)**2)
        Hs = self.getH()
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa)))
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = self.cholesky(self.Q)
        assert(self.Q_fac != -1)
        self.mvar = rqinv(self.Q).diagonal()

    def fit(self,data, r, S = None,verbose = False, grad = True, tol = 1e-5,
            par = np.array([1.1]*190)):
            #mod4: kappa(0:27), gamma1(27:54), gamma2(54:81), gamma3(81:108), rho1(108:135), rho2(135:162), rho3(162:189), sigma(189)
        self.data = data
        self.r = r
        self.S = S
        self.opt_steps = 0
        self.grad = grad
        self.verbose = verbose
        if self.grad:
            res = minimize(self.logLike, x0 = par,jac = True, method = "BFGS",tol = tol)
        else:    
            res = minimize(self.logLike, x0 = par, tol = tol)
        self.kappa = res['x'][0:27]
        self.gamma1 = res['x'][27:54]
        self.gamma2 = res['x'][54:81]
        self.gamma3 = res['x'][81:108]
        self.rho1 = res['x'][108:135]
        self.rho2 = res['x'][135:162]
        self.rho3 = res['x'][162:189]
        self.tau = res['x'][189]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def fitTo(self,simmod,dho,r,num,verbose = False, grad = True, par = np.array([1.1]*190)):
        mods = np.array(['SI','SA','NA1','NA2'])
        dhos = np.array(['100','1000','10000'])
        rs = np.array([1,10,100])
        tmp = np.load('./simulations/' + mods[simmod-1] + '-'+str(num)+".npz")
        self.data = (tmp['data']*1)[tmp['locs'+dhos[dho-1]],:(rs[r-1])]
        self.r = rs[r-1]
        self.S = np.zeros((self.n))
        self.S[tmp['locs'+dhos[dho-1]]*1] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        res = self.fit(data = self.data, r=self.r, S = self.S,verbose = verbose, grad = grad,par = par)
        np.savez('./fits/' + mods[simmod-1] + '-NA1-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res['x'], like = res['fun'],  jac = res['jac'], S = tmp['locs'+dhos[dho-1]]*1)
        return(True)

    # assertion for number of parameters
    def loadFit(self, simmod, dho, r, num, file = None):
        if file is None:
            mods = np.array(['SI','SA','NA1','NA2'])
            dhos = np.array(['100','1000','10000'])
            rs = np.array([1,10,100])
            file = './fits/' + mods[simmod-1] + '-NA1-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = np.zeros((self.grid.M*self.grid.N*self.grid.P))
        self.S[fitmod['S']*1] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        par =fitmod['par']*1
        self.kappa = par[0:27]
        self.gamma1 = par[27:54]
        self.gamma2 = par[54:81]
        self.gamma3 = par[81:108]
        self.rho1 = par[108:135]
        self.rho2 = par[135:162]
        self.rho3 = par[162:189]
        self.tau = par[189]
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
            assert(self.kappa is not None and self.gamma1 is not None and self.gamma2 is not None and self.gamma3 is not None and self.rho1 is not None and self.rho2 is not None and self.rho3 is not None and self.sigma is not None)
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
            if file.startswith("NA2-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/NA2-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs1000 = np.random.choice(np.arange(self.n), 1000, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False))
        return(True)

    def setQ(self,par = None):
        if par is None:
            assert(self.kappa is not None and self.gamma1 is not None and self.gamma2 is not None and self.gamma3 is not None and self.rho1 is not None and self.rho2 is not None and self.rho3 is not None and self.sigma is not None)
        else:
            self.kappa = par[0:27]
            self.gamma1 = par[27:54]
            self.gamma2 = par[54:81]
            self.gamma3 = par[81:108]
            self.rho1 = par[108:135]
            self.rho2 = par[135:162]
            self.rho3 = par[162:189]
            self.tau = par[189]
            self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = self.getH()
        Dk =  sparse.diags(np.exp(self.grid.evalB(par = self.kappa))) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)

    def getH(self,gamma1 = None,gamma2 = None,gamma3 = None, rho1 = None,rho2 = None,rho3 = None,d=None,var = None):
        if gamma1 is None and gamma2 is None and gamma2 is None and rho1 is None and rho2 is None and rho3 is None:
            gamma1 = self.gamma1
            gamma2 = self.gamma2
            gamma3 = self.gamma3
            rho1 = self.rho1
            rho2 = self.rho2
            rho3 = self.rho3
        if var is None:
            pg1 = np.exp(self.grid.evalBH(par = gamma1))
            pg2 = np.exp(self.grid.evalBH(par = gamma2))
            pg3 = np.exp(self.grid.evalBH(par = gamma3))
            pr1 = self.grid.evalBH(par = rho1)
            pr2 = self.grid.evalBH(par = rho2)
            pr3 = self.grid.evalBH(par = rho3)
            v = (pg2/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr2 - pr3),1 - pr1**2 + pr2**2 - pr3**2,2*(pr3*pr2 + pr1)],axis=2)
            w = (pg3/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr3 + pr2),2*(pr2*pr3 - pr1),1 - pr1**2 - pr2**2 + pr3**2],axis=2)
            H = (np.eye(3)*(np.stack([pg1,pg1,pg1],axis=2))[:,:,:,np.newaxis]) + v[:,:,:,np.newaxis]*v[:,:,np.newaxis,:] + w[:,:,:,np.newaxis]*w[:,:,np.newaxis,:]
        elif var == 0: #gamma1
            pg1 = np.exp(self.grid.evalBH(par = gamma1))
            H = np.eye(3)*(np.stack([self.grid.bsH[:,:,d]*pg1,self.grid.bsH[:,:,d]*pg1,self.grid.bsH[:,:,d]*pg1],axis=2)[:,:,:,np.newaxis])
        elif var == 1: #gamma2
            pg2 = np.exp(self.grid.evalBH(par = gamma2))
            pr1 = self.grid.evalBH(par = rho1)
            pr2 = self.grid.evalBH(par = rho2)
            pr3 = self.grid.evalBH(par = rho3)
            v = (pg2/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr2 - pr3),1-pr1**2 + pr2**2 - pr3**2,2*(pr3*pr2 + pr1)],axis=2)
            H = 2*self.grid.bsH[:,:,d]*(v[:,:,:,np.newaxis]*v[:,:,np.newaxis,:])
        elif var == 2: #gamma3
            pg3 = np.exp(self.grid.evalBH(par = gamma3))
            pr1 = self.grid.evalBH(par = rho1)
            pr2 = self.grid.evalBH(par = rho2)
            pr3 = self.grid.evalBH(par = rho3)
            w = (pg3/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr3 + pr2),2*(pr2*pr3 - pr1),1 - pr1**2 - pr2**2 + pr3**2],axis=2)
            H = 2*self.grid.bsH[:,:,d]*(w[:,:,:,np.newaxis]*w[:,:,np.newaxis,:])
        elif var == 3: #rho1
            pg2 = np.exp(self.grid.evalBH(par = gamma2))
            pg3 = np.exp(self.grid.evalBH(par = gamma3))
            pr1 = self.grid.evalBH(par = rho1)
            pr2 = self.grid.evalBH(par = rho2)
            pr3 = self.grid.evalBH(par = rho3)
            v = (pg2/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr2 - pr3),1-pr1**2 + pr2**2 - pr3**2,2*(pr3*pr2 + pr1)],axis=2)
            w = (pg3/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr3 + pr2),2*(pr2*pr3 - pr1),1 - pr1**2 - pr2**2 + pr3**2],axis=2)
            dv = self.grid.bsH[:,:,d][:,:,np.newaxis](2/(1+pr1**2+pr2**2+pr3**2))[:,:,np.newaxis]*(pg2[:,:,np.newaxis]*np.stack([pr2,-pr1,np.ones((self.n,6))],axis=2)-pr1[:,:,np.newaxis]*v)
            dw = self.grid.bsH[:,:,d][:,:,np.newaxis](2/(1+pr1**2+pr2**2+pr3**2))[:,:,np.newaxis]*(pg3[:,:,np.newaxis]*np.stack([pr3,-np.ones((self.n,6)),-pr1],axis=2)-pr1[:,:,np.newaxis]*w)
            H =2*dv[:,:,:,np.newaxis]*v[:,:,np.newaxis,:] + 2*dw[:,:,:,np.newaxis]*w[:,:,np.newaxis,:]
        elif var == 4: #rho2
            pg2 = np.exp(self.grid.evalBH(par = gamma2))
            pg3 = np.exp(self.grid.evalBH(par = gamma3))
            pr1 = self.grid.evalBH(par = rho1)
            pr2 = self.grid.evalBH(par = rho2)
            pr3 = self.grid.evalBH(par = rho3)
            v = (pg2/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr2 - pr3),1-pr1**2 + pr2**2 - pr3**2,2*(pr3*pr2 + pr1)],axis=2)
            w = (pg3/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr3 + pr2),2*(pr2*pr3 - pr1),1 - pr1**2 - pr2**2 + pr3**2],axis=2)
            dv = self.grid.bsH[:,:,d][:,:,np.newaxis]*(2/(1+pr1**2+pr2**2+pr3**2))[:,:,np.newaxis]*(pg2[:,:,np.newaxis]*np.stack([pr1,pr2,pr3],axis=2)-pr2[:,:,np.newaxis]*v)
            dw = self.grid.bsH[:,:,d][:,:,np.newaxis]*(2/(1+pr1**2+pr2**2+pr3**2))[:,:,np.newaxis]*(pg3[:,:,np.newaxis]*np.stack([np.ones((self.n,6)),pr3,-pr2],axis=2)-pr2[:,:,np.newaxis]*w)
            H =2*dv[:,:,:,np.newaxis]*v[:,:,np.newaxis,:] + 2*dw[:,:,:,np.newaxis]*w[:,:,np.newaxis,:]
        elif var == 5: #rho3
            pg2 = np.exp(self.grid.evalBH(par = gamma2))
            pg3 = np.exp(self.grid.evalBH(par = gamma3))
            pr1 = self.grid.evalBH(par = rho1)
            pr2 = self.grid.evalBH(par = rho2)
            pr3 = self.grid.evalBH(par = rho3)
            v = (pg2/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr2 - pr3),1-pr1**2 + pr2**2 - pr3**2,2*(pr3*pr2 + pr1)],axis=2)
            w = (pg3/(1+pr1**2+pr2**2+pr3**3))[:,:,np.newaxis]*np.stack([2*(pr1*pr3 + pr2),2*(pr2*pr3 - pr1),1 - pr1**2 - pr2**2 + pr3**2],axis=2)
            dv = self.grid.bsH[:,:,d][:,:,np.newaxis](2/(1+pr1**2+pr2**2+pr3**2))[:,:,np.newaxis]*(pg2[:,:,np.newaxis]*np.stack([-np.ones((self.n,6)),-pr3,pr2],axis=2)-pr3[:,:,np.newaxis]*v)
            dw = self.grid.bsH[:,:,d][:,:,np.newaxis](2/(1+pr1**2+pr2**2+pr3**2))[:,:,np.newaxis]*(pg3[:,:,np.newaxis]*np.stack([pr1,pr2,pr3],axis=2)-pr3[:,:,np.newaxis]*w)
            H =2*dv[:,:,:,np.newaxis]*v[:,:,np.newaxis,:] + 2*dw[:,:,:,np.newaxis]*w[:,:,np.newaxis,:]
        return(H)


    def cholesky(self,Q):
        try: 
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)

    def logLike(self, par):
        #mod4: kappa(0:27), gamma1(27:54), gamma2(54:81), gamma3(81:108), rho1(108:135), rho2(135:162), rho3(162:189), sigma(189)
        data  = self.data
        Hs = self.getH(gamma1 = par[27:54],gamma2 = par[54:81], gamma3 = par[81:108], rho1 = par[108:135],rho2=par[135:162],rho3=par[162:189]) 
        lkappa = self.grid.evalB(par = par[0:27])
        Dk =  sparse.diags(np.exp(lkappa)) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        Q = A_mat.transpose()@self.iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[189])
        Q_fac = self.cholesky(Q)
        Q_c_fac= self.cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if self.grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[189]))
        if self.r == 1:
            data = data.reshape(data.shape[0],1)
            mu_c = mu_c.reshape(mu_c.shape[0],1)
        if self.grad:
            Qinv = rqinv(Q)
            Qcinv = rqinv(Q_c)

            g_kappa = np.zeros(27)
            g_gamma1 = np.zeros(27)
            g_gamma2 = np.zeros(27)
            g_gamma3 = np.zeros(27)
            g_rho1 = np.zeros(27)
            g_rho2 = np.zeros(27)
            g_rho3 = np.zeros(27)

            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[189]/2 - 1/2*Q_c_fac.logdet()*self.r
            g_noise = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[189])).diagonal().sum()*self.r
            for i in range(27):
                Hs_par = self.getH(gamma1 = par[27:54],gamma2 = par[54:81], gamma3 = par[81:108], rho1 = par[108:135],rho2=par[135:162],rho3=par[162:189],d=i,var = 0) 
                A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
                Ah_par = sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
                A_par = - Ah_par
                Q_gamma1 = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par

                Hs_par = self.getH(gamma1 = par[27:54],gamma2 = par[54:81], gamma3 = par[81:108], rho1 = par[108:135],rho2=par[135:162],rho3=par[162:189],d=i,var = 1)
                A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
                Ah_par = sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
                A_par = - Ah_par
                Q_gamma2 = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par

                Hs_par = self.getH(gamma1 = par[27:54],gamma2 = par[54:81], gamma3 = par[81:108], rho1 = par[108:135],rho2=par[135:162],rho3=par[162:189],d=i,var = 2)
                A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
                Ah_par = sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
                A_par = - Ah_par
                Q_gamma3 = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par

                Hs_par = self.getH(gamma1 = par[27:54],gamma2 = par[54:81], gamma3 = par[81:108], rho1 = par[108:135],rho2=par[135:162],rho3=par[162:189],d=i,var = 3)
                A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
                Ah_par = sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
                A_par = - Ah_par
                Q_rho1 = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par

                Hs_par = self.getH(gamma1 = par[27:54],gamma2 = par[54:81], gamma3 = par[81:108], rho1 = par[108:135],rho2=par[135:162],rho3=par[162:189],d=i,var = 4)
                A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
                Ah_par = sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
                A_par = - Ah_par
                Q_rho2 = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par

                Hs_par = self.getH(gamma1 = par[27:54],gamma2 = par[54:81], gamma3 = par[81:108], rho1 = par[108:135],rho2=par[135:162],rho3=par[162:189],d=i,var = 5)
                A_H_par = AH(self.grid.M,self.grid.N,self.grid.P,Hs_par,self.grid.hx,self.grid.hy,self.grid.hz)
                Ah_par = sparse.csc_matrix((A_H_par.Val(), (A_H_par.Row(), A_H_par.Col())), shape=(self.n, self.n))
                A_par = - Ah_par
                Q_rho3 = A_par.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_par

                Dk2 = sparse.diags(self.grid.bs[:,i]*np.exp(lkappa))
                A_par = self.Dv@Dk2
                Q_kappa = A_par.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_par
                                                                   
                g_kappa[i] = 1/2*((Qinv - Qcinv)@Q_kappa).diagonal().sum()*self.r
                g_gamma1[i] = 1/2*((Qinv - Qcinv)@Q_gamma1).diagonal().sum()*self.r
                g_gamma2[i] = 1/2*((Qinv - Qcinv)@Q_gamma2).diagonal().sum()*self.r
                g_gamma3[i] = 1/2*((Qinv - Qcinv)@Q_gamma3).diagonal().sum()*self.r
                g_rho1[i] = 1/2*((Qinv - Qcinv)@Q_rho1).diagonal().sum()*self.r
                g_rho2[i] = 1/2*((Qinv - Qcinv)@Q_rho2).diagonal().sum()*self.r
                g_rho3[i] = 1/2*((Qinv - Qcinv)@Q_rho3).diagonal().sum()*self.r
                for j in range(self.r): 
                    g_kappa[i] = g_kappa[i] + (- 1/2*mu_c[:,j].transpose()@Q_kappa@mu_c[:,j])
                    g_gamma1[i] = g_gamma1[i] + (- 1/2*mu_c[:,j].transpose()@Q_gamma1@mu_c[:,j])
                    g_gamma2[i] = g_gamma2[i] + (- 1/2*mu_c[:,j].transpose()@Q_gamma2@mu_c[:,j])
                    g_gamma3[i] = g_gamma3[i] + (- 1/2*mu_c[:,j].transpose()@Q_gamma3@mu_c[:,j])
                    g_rho1[i] = g_rho1[i] + (- 1/2*mu_c[:,j].transpose()@Q_rho1@mu_c[:,j])
                    g_rho2[i] = g_rho2[i] + (- 1/2*mu_c[:,j].transpose()@Q_rho2@mu_c[:,j])
                    g_rho3[i] = g_rho3[i] + (- 1/2*mu_c[:,j].transpose()@Q_rho3@mu_c[:,j])
                    if i == 0:
                        g_noise = g_noise + (- 1/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j] - self.S@mu_c[:,j])*np.exp(par[189]))
                        like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[189])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = - like/(self.S.shape[0]*self.r)
            jac = np.hstack([g_kappa,g_gamma1,g_gamma2,g_gamma3,g_rho1,g_rho2,g_rho3,g_noise])
            jac = - jac/(self.S.shape[0]*self.r)
            self.opt_steps = self.opt_steps + 1
            self.like = like
            self.jac = jac
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like))#, "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3 = %2.2f"%np.exp(par[1]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[2])))
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*self.r*par[189]/2 - 1/2*Q_c_fac.logdet()*self.r
            for j in range(self.r):
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[189])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = -like/(self.S.shape[0]*self.r)
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like))#, "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3 = %2.2f"%np.exp(par[1]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[2])))
            return(like)