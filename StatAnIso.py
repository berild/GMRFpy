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


class StatAnIso:
    #mod2: kappa(0), gamma1(1), gamma2(2), gamma3(3), rho1(4), rho2(5), rho3(6), sigma(7)
    def __init__(self,grid=None,par=None):
        assert par is None or (par.size==8 if par is not None else False)
        self.grid = Grid() if grid is None else grid
        self.V = self.grid.hx*self.grid.hy*self.grid.hz
        self.n = self.grid.M*self.grid.N*self.grid.P
        self.kappa = 0.5 if par is None else par[0]
        self.gamma1 = 0.5 if par is None else par[1]
        self.gamma2 = 0.5 if par is None else par[2]
        self.gamma3 = 0.5 if par is None else par[3]
        self.rho1 = 0.5 if par is None else par[4]
        self.rho2 = 0.5 if par is None else par[5]
        self.rho3 = 0.5 if par is None else par[6]
        self.tau = 0.5 if par is None else par[7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        self.Dv = self.V*sparse.eye(self.n)
        self.iDv = 1/self.V*sparse.eye(self.n)
        self.v, self.w = self.getVW(np.array([self.gamma2,self.gamma3,self.rho1,self.rho2,self.rho3]))
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
        self.jac = np.array([-100]*8)

    def load(self):
        simmod = np.load("./simmodels/SA.npz")
        self.kappa = simmod['kappa']*1
        self.gamma1 = simmod['gamma1']*1
        self.gamma2 = simmod['gamma2']*1
        self.gamma3 = simmod['gamma3']*1
        self.rho1 = simmod['rho1']*1
        self.rho2 = simmod['rho2']*1
        self.rho3 = simmod['rho3']*1
        self.sigma = simmod['sigma']*1
        self.tau = np.log(1/np.exp(self.sigma)**2)
        self.v, self.w = self.getVW(np.array([self.gamma2,self.gamma3,self.rho1,self.rho2,self.rho3]))
        Hs = np.exp(self.gamma1)*np.eye(3) + self.v[:,np.newaxis]*self.v[np.newaxis,:]  + self.w[:,np.newaxis]*self.w[np.newaxis,:]  + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags([np.exp(self.kappa)]*(self.n)) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = self.cholesky(self.Q)
        assert(self.Q_fac != -1)
        self.mvar = rqinv(self.Q).diagonal()
        
    def fit(self,data, r, S = None,verbose = False, grad = True, tol = 1e-5,
            par = np.array([np.log(1.1),np.log(1.1),np.log(1.2),np.log(1.3),0.2,0.2,0.2,5])):
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
        self.kappa = res['x'][0]
        self.gamma1 = res['x'][1]
        self.gamma2 = res['x'][2]
        self.gamma3 = res['x'][3]
        self.rho1 = res['x'][4]
        self.rho2 = res['x'][5]
        self.rho3 = res['x'][6]
        self.tau = res['x'][7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def fitTo(self,simmod,dho,r,num,verbose = False, grad = True, par = np.log(np.array([1.1,1.1,1.1]))):
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
        np.savez('./fits/' + mods[simmod-1] + '-SA-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res['x'], like = res['fun'],  jac = res['jac'], S = tmp['locs'+dhos[dho-1]]*1)
        return(True)

    # maybe add some assertions
    def loadFit(self, simmod, dho, r, num, file = None):
        if file is None:
            mods = np.array(['SI','SA','NA1','NA2'])
            dhos = np.array(['100','1000','10000'])
            rs = np.array([1,10,100])
            file = './fits/' + mods[simmod-1] + '-SA-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = np.zeros((self.grid.M*self.grid.N*self.grid.P))
        self.S[fitmod['S']*1] = 1
        self.S = sparse.diags(self.S)
        self.S =  delete_rows_csr(self.S.tocsr(),np.where(self.S.diagonal() == 0))
        par = fitmod['par']*1
        self.kappa = par[0]
        self.gamma1 = par[1]
        self.gamma2 = par[2]
        self.gamma3 = par[3]
        self.rho1 = par[4]
        self.rho2 = par[5]
        self.rho3 = par[6]
        self.tau = par[7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        Hs = np.exp(self.gamma1)*np.eye(3) + self.v[:,np.newaxis]*self.v[np.newaxis,:]  + self.w[:,np.newaxis]*self.w[np.newaxis,:]  + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags([np.exp(self.kappa)]*self.n) 
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
            if file.startswith("SA-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/SA-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs1000 = np.random.choice(np.arange(self.n), 1000, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False))
        return(True)

    def setQ(self,par = None):
        if par is None:
            assert(self.kappa is not None and self.gamma1 is not None and self.gamma2 is not None and self.gamma3 is not None and self.rho1 is not None and self.rho2 is not None and self.rho3 is not None and self.sigma is not None)
        else:
            self.kappa = par[0]
            self.gamma1 = par[1]
            self.gamma2 = par[2]
            self.gamma3 = par[3]
            self.rho1 = par[4]
            self.rho2 = par[5]
            self.rho3 = par[6]
            self.sigma = par[7]
            self.tau = np.log(1/np.exp(self.sigma)**2)
        self.v, self.w = self.getVW(np.array([self.gamma2,self.gamma3,self.rho1,self.rho2,self.rho3]))
        Hs = np.exp(self.gamma1)*np.eye(3) + self.v[:,np.newaxis]*self.v[np.newaxis,:]  + self.w[:,np.newaxis]*self.w[np.newaxis,:]  + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags([np.exp(self.kappa)]*self.n) 
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
            

    def getVW(self,par):
        assert(par.size == 5)
        vs = np.exp(par[0])*np.array([2*(par[2]*par[3] - par[4]),1 - par[2]**2 + par[3]**2 - par[4]**2,2*(par[3]*par[4] + par[2])])/(1 + par[2]**2 + par[3]**2 + par[4]**2)
        ws = np.exp(par[1])*np.array([2*(par[2]*par[4] + par[3]),2*(par[3]*par[4] - par[2]),1 - par[2]**2 - par[3]**2 + par[4]**2])/(1 + par[2]**2 + par[3]**2 + par[4]**2)
        return((vs,ws))

    def getH(self,par,d=None):
        v = np.exp(par[1])*np.array([2*(par[3]*par[4] - par[5]),1 - par[3]**2 + par[4]**2 - par[5]**2,2*(par[5]*par[4] + par[3])])/(1 + par[3]**2 + par[4]**2 + par[5]**2)
        w = np.exp(par[2])*np.array([2*(par[3]*par[5] + par[4]),2*(par[4]*par[5] - par[3]),1 - par[3]**2 - par[4]**2 + par[5]**2])/(1 + par[3]**2 + par[4]**2 + par[5]**2)
        if d is None:
            H = np.exp(par[0])*np.eye(3) + v[:,np.newaxis]*v[np.newaxis,:]  + w[:,np.newaxis]*w[np.newaxis,:]  + np.zeros((self.n,6,3,3))
        elif d == 0:
            H = np.exp(par[1])*np.eye(3) + np.zeros((self.n,6,3,3))
        elif d == 1:
            H =  2*v[:,np.newaxis]*v[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 2:
            H =  2*w[:,np.newaxis]*w[np.newaxis,:]  + np.zeros((self.n,6,3,3))
        elif d == 3:
            dv = 2/(1+par[3]**2+par[4]**2+par[5]**2)*(np.array([par[4],-par[3],1])*np.exp(par[1])-par[3]*v)
            dw = 2/(1+par[3]**2+par[4]**2+par[5]**2)*(np.array([par[5],-1,-par[3]])*np.exp(par[2])-par[3]*w)
            H = 2*v[:,np.newaxis]*dv[np.newaxis,:] + 2*w[:,np.newaxis]*dw[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 4:
            dv = 2/(1+par[3]**2+par[4]**2+par[5]**2)*(np.array([par[3],par[4],par[5]])*np.exp(par[1])-par[4]*v)
            dw = 2/(1+par[3]**2+par[4]**2+par[5]**2)*(np.array([1,par[5],-par[4]])*np.exp(par[2])-par[4]*w)
            H = 2*v[:,np.newaxis]*dv[np.newaxis,:] + 2*w[:,np.newaxis]*dw[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 5:
            dv = 2/(1+par[3]**2+par[4]**2+par[5]**2)*(np.array([-1,-par[3],par[5]])*np.exp(par[1])-par[5]*v)
            dw = 2/(1+par[3]**2+par[4]**2+par[5]**2)*(np.array([par[3],par[4],par[5]])*np.exp(par[2])-par[5]*w)
            H = 2*v[:,np.newaxis]*dv[np.newaxis,:] + 2*w[:,np.newaxis]*dw[np.newaxis,:] + np.zeros((self.n,6,3,3))
        return(H)

    def logLike(self, par):
        data  = self.data
        Hs = self.getH(par[1:7])
        Dk =  np.exp(par[0])*sparse.eye(self.n) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
        A_mat = self.Dv@Dk - Ah
        Q = A_mat.transpose()@self.iDv@A_mat
        Q_c = Q + self.S.transpose()@self.S*np.exp(par[7])
        Q_fac = self.cholesky(Q)
        Q_c_fac= self.cholesky(Q_c)
        if (Q_fac == -1) or (Q_c_fac == -1):
            if self.grad:
                return((self.like,self.jac))
            else:
                return(self.like)
        mu_c = Q_c_fac.solve_A(self.S.transpose()@data*np.exp(par[7]))
        if self.r == 1:
            data = data.reshape(data.shape[0],1)
            mu_c = mu_c.reshape(mu_c.shape[0],1)
        if self.grad:
            Qinv = rqinv(Q)
            Qcinv = rqinv(Q_c)
            # these are missing
            # implement a function for Hs for simplicity
            Hs_gamma1 = self.getH(par[1:7],d=0)
            Hs_gamma2 = self.getH(par[1:7],d=1)
            Hs_gamma3 = self.getH(par[1:7],d=2)
            Hs_rho1 = self.getH(par[1:7],d=3)
            Hs_rho2 = self.getH(par[1:7],d=4)
            Hs_rho3 = self.getH(par[1:7],d=5)

            A_H_gamma1 = AH(self.grid.M,self.grid.N,self.grid.P,Hs_gamma1,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_gamma2 = AH(self.grid.M,self.grid.N,self.grid.P,Hs_gamma2,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_gamma3 = AH(self.grid.M,self.grid.N,self.grid.P,Hs_gamma3,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_rho1 = AH(self.grid.M,self.grid.N,self.grid.P,Hs_rho1,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_rho2 = AH(self.grid.M,self.grid.N,self.grid.P,Hs_rho2,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_rho3 = AH(self.grid.M,self.grid.N,self.grid.P,Hs_rho3,self.grid.hx,self.grid.hy,self.grid.hz)

            Ah_gamma1 = sparse.csc_matrix((A_H_gamma1.Val(), (A_H_gamma1.Row(), A_H_gamma1.Col())), shape=(self.n, self.n))
            Ah_gamma2 = sparse.csc_matrix((A_H_gamma2.Val(), (A_H_gamma2.Row(), A_H_gamma2.Col())), shape=(self.n, self.n))
            Ah_gamma3 = sparse.csc_matrix((A_H_gamma3.Val(), (A_H_gamma3.Row(), A_H_gamma3.Col())), shape=(self.n, self.n))
            Ah_rho1 = sparse.csc_matrix((A_H_rho1.Val(), (A_H_rho1.Row(), A_H_rho1.Col())), shape=(self.n, self.n))
            Ah_rho2 = sparse.csc_matrix((A_H_rho2.Val(), (A_H_rho2.Row(), A_H_rho2.Col())), shape=(self.n, self.n))
            Ah_rho3 = sparse.csc_matrix((A_H_rho3.Val(), (A_H_rho3.Row(), A_H_rho3.Col())), shape=(self.n, self.n))

            A_kappa = Dk@self.Dv
            A_gamma1 = - Ah_gamma1
            A_gamma2 = - Ah_gamma2
            A_gamma3 = - Ah_gamma3
            A_rho1 = - Ah_rho1
            A_rho2 = - Ah_rho2
            A_rho3 = - Ah_rho3

            Q_kappa = A_kappa.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_kappa
            Q_gamma1 = A_gamma1.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_gamma1
            Q_gamma2 = A_gamma2.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_gamma2
            Q_gamma3 = A_gamma3.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_gamma3
            Q_rho1 = A_rho1.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_rho1
            Q_rho2 = A_rho2.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_rho2
            Q_rho3 = A_rho3.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_rho3

            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*par[7]*self.r/2 - 1/2*Q_c_fac.logdet()*self.r
            g_kappa = 1/2*((Qinv - Qcinv)@Q_kappa).diagonal().sum()*self.r
            g_gamma1 = 1/2*((Qinv - Qcinv)@Q_gamma1).diagonal().sum()*self.r
            g_gamma2 = 1/2*((Qinv - Qcinv)@Q_gamma2).diagonal().sum()*self.r
            g_gamma3 = 1/2*((Qinv - Qcinv)@Q_gamma3).diagonal().sum()*self.r
            g_rho1 = 1/2*((Qinv - Qcinv)@Q_rho1).diagonal().sum()*self.r
            g_rho2 = 1/2*((Qinv - Qcinv)@Q_rho2).diagonal().sum()*self.r
            g_rho3 = 1/2*((Qinv - Qcinv)@Q_rho3).diagonal().sum()*self.r
            g_noise = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[7])).diagonal().sum()*self.r

            for j in range(self.r): # Maybe make a better version than this for loop possibly need to account for dimension 0
                g_kappa = g_kappa + (- 1/2*mu_c[:,j].transpose()@Q_kappa@mu_c[:,j])
                g_gamma1 = g_gamma1 + (- 1/2*mu_c[:,j].transpose()@Q_gamma1@mu_c[:,j])
                g_gamma2 = g_gamma2 + (- 1/2*mu_c[:,j].transpose()@Q_gamma2@mu_c[:,j])
                g_gamma3 = g_gamma3 + (- 1/2*mu_c[:,j].transpose()@Q_gamma3@mu_c[:,j])
                g_rho1 = g_rho1 + (- 1/2*mu_c[:,j].transpose()@Q_rho1@mu_c[:,j])
                g_rho2 = g_rho2 + (- 1/2*mu_c[:,j].transpose()@Q_rho2@mu_c[:,j])
                g_rho3 = g_rho3 + (- 1/2*mu_c[:,j].transpose()@Q_rho3@mu_c[:,j])
                g_noise = g_noise + (- 1/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j] - self.S@mu_c[:,j])*np.exp(par[7]))
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[7])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = -like/(self.r * self.S.shape[0])
            jac = - np.array([g_kappa,g_gamma1,g_gamma2,g_gamma3,g_rho1,g_rho2,g_rho3,g_noise])/(self.r * self.S.shape[0])
            self.opt_steps = self.opt_steps + 1
            self.like = like
            self.jac = jac
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3_1 = %2.2f"%np.exp(par[1]),"\u03B3_2 = %2.2f"%np.exp(par[2]),"\u03B3_3 = %2.2f"%np.exp(par[3]),
                "\u03C1_1 = %2.2f"%(par[4]),"\u03C1_2 = %2.2f"%(par[5]),"\u03C1_3 = %2.2f"%(par[6]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[7])))
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet() + self.S.shape[0]*par[7]/2 - 1/2*Q_c_fac.logdet() 
            for j in range(self.r):
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[7])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))/self.r
            like = -like/(self.r * self.S.shape[0])
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3_1 = %2.2f"%np.exp(par[1]),"\u03B3_2 = %2.2f"%np.exp(par[7]),"\u03B3_3 = %2.2f"%np.exp(par[3]),
                "\u03C1_1 = %2.2f"%(par[4]),"\u03C1_2 = %2.2f"%(par[5]),"\u03C1_3 = %2.2f"%(par[6]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[7])))
            return(like)

    