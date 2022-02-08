import sys, getopt
import numpy as np
from scipy import sparse
from ah3d2 import AH
from sksparse.cholmod import cholesky
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
inla = importr("INLA")
robj.r('inla.setOption("smtp" = "pardiso", pardiso.license = "./pardiso.lic")')
from scipy.optimize import minimize
import os
from grid import Grid
import time

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
        self.gammaX = 0.5 if par is None else par[1]
        self.gammaY = 0.5 if par is None else par[2]
        self.gammaZ = 0.5 if par is None else par[3]
        self.vx = 0.5 if par is None else par[4]
        self.vy = 0.5 if par is None else par[5]
        self.vz = 0.5 if par is None else par[6]
        self.tau = 0.5 if par is None else par[7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        self.Dv = self.V*sparse.eye(self.n)
        self.iDv = 1/self.V*sparse.eye(self.n)
        self.v = np.array([self.vx,self.vy,self.vz])
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
        self.loaded = False

    def load(self,simple = False):
        if self.loaded:
            return
        simmod = np.load("./simmodels/SA1.npz")
        self.kappa = simmod['kappa']*1
        self.gammaX = simmod['gammaX']*1
        self.gammaY = simmod['gammaY']*1
        self.gammaZ = simmod['gammaZ']*1
        self.vx = simmod['vx']*1
        self.vy = simmod['vy']*1
        self.vz = simmod['vz']*1
        self.sigma = simmod['sigma']*1
        self.tau = np.log(1/np.exp(self.sigma)**2)
        if not simple:
            self.v = np.array([self.vx,self.vy,self.vz])
            Hs = np.diag(np.exp([self.gammaX,self.gammaY,self.gammaZ])) + self.v[:,np.newaxis]*self.v[np.newaxis,:] + np.zeros((self.n,6,3,3))
            Dk =  sparse.diags([np.exp(self.kappa)]*(self.n)) 
            A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
            Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n, self.n))
            A_mat = self.Dv@Dk - Ah
            self.Q = A_mat.transpose()@self.iDv@A_mat
            self.Q_fac = self.cholesky(self.Q)
            assert(self.Q_fac != -1)
            self.mvar = rqinv(self.Q).diagonal()
            self.loaded = True
        
    def fit(self,data, r, S = None,par = None,verbose = False, grad = True):
        if par is None:
            par = np.array([-1.5,0.5,0.5,-0.5,0.5,0.5,0.5,2])
        assert S is not None
        self.data = data
        self.r = r
        self.S = S
        self.opt_steps = 0
        self.grad = grad
        self.verbose = verbose
        if self.grad:
            res = minimize(self.logLike, x0 = par,jac = True, method = "BFGS")
            os.write(1, b'after \n')
        else:    
            res = minimize(self.logLike, x0 = par, tol = 1e-3)
        self.kappa = res['x'][0]
        self.gammaX = res['x'][1]
        self.gammaY = res['x'][2]
        self.gammaZ = res['x'][3]
        self.vx = res['x'][4]
        self.vy = res['x'][5]
        self.vz = res['x'][6]
        self.tau = res['x'][7]
        self.sigma = res['x'][7] #np.log(np.sqrt(1/np.exp(self.tau)))
        return(res)

    def fitTo(self,simmod,dho,r,num,verbose = False, grad = True, par = None):
        if par is None:
            par = np.array([-1.5,0.5,0.5,-0.5,0.5,0.5,0.5,2])
        mods = np.array(['SI','SA','NA1','NA2','SA1'])
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
        print(res['x'])
        np.savez(file = './fits/' + mods[simmod-1] + '-SA1-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz', par = res['x'], S = np.sort(tmp['locs'+dhos[dho-1]]*1))
        return(True)
        

    # maybe add some assertions
    def loadFit(self, simmod, dho, r, num, file = None):
        if file is None:
            mods = np.array(['SI','SA','NA1','NA2','SA1'])
            dhos = np.array(['100','10000','27000'])
            rs = np.array([1,10,100])
            file = './fits/' + mods[simmod-1] + '-SA1-dho' + dhos[dho-1] + '-r' + str(rs[r-1]) + '-' + str(num) +'.npz'
            print(file)
        fitmod = np.load(file)
        self.S = fitmod['S']*1
        par = fitmod['par']*1
        self.kappa = par[0]
        self.gammaX = par[1]
        self.gammaY = par[2]
        self.gammaZ = par[3]
        self.vx = par[4]
        self.vy = par[5]
        self.vz = par[6]
        self.tau = par[7]
        self.sigma = np.log(np.sqrt(1/np.exp(self.tau)))
        self.v = np.array([self.vx,self.vy,self.vz])
        Hs = np.diag(np.exp([self.gammaX,self.gammaY,self.gammaZ])) + self.v[:,np.newaxis]*self.v[np.newaxis,:] + np.zeros((self.n,6,3,3))
        Dk =  sparse.diags([np.exp(self.kappa)]*self.n) 
        A_H = AH(self.grid.M,self.grid.N,self.grid.P,Hs,self.grid.hx,self.grid.hy,self.grid.hz)
        Ah = sparse.csc_matrix((A_H.Val(), (A_H.Row(), A_H.Col())), shape=(self.n,self.n))
        A_mat = self.Dv@Dk - Ah
        self.Q = A_mat.transpose()@self.iDv@A_mat
        self.Q_fac = cholesky(self.Q)
        self.mvar = rqinv(self.Q).diagonal()

    def getPars(self):
        return(np.hstack([self.kappa,self.gammaX,self.gammaY,self.gammaZ,self.vx,self.vy,self.vz,self.tau]))

    def sample(self,n = 1, par = None):
        if par is None:
            assert(self.kappa is not None and self.gammaX is not None and self.gammaY is not None and self.gammaZ is not None and self.vx is not None and self.vy is not None and self.vz is not None and self.sigma is not None)
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
            if file.startswith("SA1-"):
                mods.append(int(file.split("-")[1].split(".")[0]))
        if not mods:
            num = 1
        else:
            num = max(mods) + 1 
        self.data = self.sample(n=100)
        np.savez('./simulations/SA1-'+ str(num) +'.npz', data = self.data, locs100 = np.random.choice(np.arange(self.n), 100, replace = False), locs10000 = np.random.choice(np.arange(self.n), 10000, replace = False), locs27000 = np.arange(self.n))
        return(True)

    def setQ(self,par = None):
        if par is None:
            assert(self.kappa is not None and self.gammaX is not None and self.gammaY is not None and self.gammaZ is not None and self.vx is not None and self.vy is not None and self.vz is not None and self.sigma is not None)
        else:
            self.kappa = par[0]
            self.gammaX = par[1]
            self.gammaY = par[2]
            self.gammaZ = par[3]
            self.vx = par[4]
            self.vy = par[5]
            self.vz = par[6]
            self.sigma = par[7]
            self.tau = np.log(1/np.exp(self.sigma)**2)
        self.v = np.array([self.vx,self.vy,self.vz])
        Hs = np.diag(np.exp([self.gammaX,self.gammaY,self.gammaZ])) + self.v[:,np.newaxis]*self.v[np.newaxis,:] + np.zeros((self.n,6,3,3))
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
            
            
    def getH(self,par,d=None):
        v = np.array([par[3],par[4],par[5]])
        if d is None:
            Hs = np.diag(np.exp([par[0],par[1],par[2]])) + v[:,np.newaxis]*v[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 0:
            H = np.diag([np.exp(par[0]),0,0]) + np.zeros((self.n,6,3,3))
        elif d == 1:
            H = np.diag([0,np.exp(par[1]),0]) + np.zeros((self.n,6,3,3))
        elif d == 2:
            H = np.diag([0,0,np.exp(par[2])]) + np.zeros((self.n,6,3,3))
        elif d == 3:
            dv = np.array([par[3],0,0])
            H = 2*v[:,np.newaxis]*dv[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 4:
            dv = np.array([0,par[4],0])
            H = 2*v[:,np.newaxis]*dv[np.newaxis,:] + np.zeros((self.n,6,3,3))
        elif d == 5:
            dv = np.array([0,0,par[5]])
            H = 2*v[:,np.newaxis]*dv[np.newaxis,:] + np.zeros((self.n,6,3,3))
        return(H)

    def logLike(self, par):
        #os.write(1, b'begining  \n')
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
            Hs_gammaX = self.getH(par[1:7],d=0)
            Hs_gammaY = self.getH(par[1:7],d=1)
            Hs_gammaZ = self.getH(par[1:7],d=2)
            Hs_vx = self.getH(par[1:7],d=3)
            Hs_vy = self.getH(par[1:7],d=4)
            Hs_vz = self.getH(par[1:7],d=5)

            A_H_gammaX = AH(self.grid.M,self.grid.N,self.grid.P,Hs_gammaX,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_gammaY = AH(self.grid.M,self.grid.N,self.grid.P,Hs_gammaY,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_gammaZ = AH(self.grid.M,self.grid.N,self.grid.P,Hs_gammaZ,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_vx = AH(self.grid.M,self.grid.N,self.grid.P,Hs_vx,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_vz = AH(self.grid.M,self.grid.N,self.grid.P,Hs_vy,self.grid.hx,self.grid.hy,self.grid.hz)
            A_H_vy = AH(self.grid.M,self.grid.N,self.grid.P,Hs_vz,self.grid.hx,self.grid.hy,self.grid.hz)

            A_gammaX = - sparse.csc_matrix((A_H_gammaX.Val(), (A_H_gammaX.Row(), A_H_gammaX.Col())), shape=(self.n, self.n))
            A_gammaY = - sparse.csc_matrix((A_H_gammaY.Val(), (A_H_gammaY.Row(), A_H_gammaY.Col())), shape=(self.n, self.n))
            A_gammaZ = - sparse.csc_matrix((A_H_gammaZ.Val(), (A_H_gammaZ.Row(), A_H_gammaZ.Col())), shape=(self.n, self.n))
            A_vx = - sparse.csc_matrix((A_H_vx.Val(), (A_H_vx.Row(), A_H_vx.Col())), shape=(self.n, self.n))
            A_vy = - sparse.csc_matrix((A_H_vy.Val(), (A_H_vy.Row(), A_H_vy.Col())), shape=(self.n, self.n))
            A_vz = - sparse.csc_matrix((A_H_vz.Val(), (A_H_vz.Row(), A_H_vz.Col())), shape=(self.n, self.n))

            A_kappa = Dk@self.Dv

            Q_kappa = A_kappa.transpose()@self.iDv@A_mat + A_mat.transpose()@self.iDv@A_kappa
            Q_gammaX = A_gammaX.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_gammaX
            Q_gammaY = A_gammaY.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_gammaY
            Q_gammaZ = A_gammaZ.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_gammaZ
            Q_vx = A_vx.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_vx
            Q_vy = A_vy.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_vy
            Q_vz = A_vz.transpose()@self.iDv@A_mat +  A_mat.transpose()@self.iDv@A_vz

            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*par[7]*self.r/2 - 1/2*Q_c_fac.logdet()*self.r
            g_kappa = 1/2*((Qinv - Qcinv)@Q_kappa).diagonal().sum()*self.r
            g_gammaX = 1/2*((Qinv - Qcinv)@Q_gammaX).diagonal().sum()*self.r
            g_gammaY = 1/2*((Qinv - Qcinv)@Q_gammaY).diagonal().sum()*self.r
            g_gammaZ = 1/2*((Qinv - Qcinv)@Q_gammaZ).diagonal().sum()*self.r
            g_vx = 1/2*((Qinv - Qcinv)@Q_vx).diagonal().sum()*self.r
            g_vy = 1/2*((Qinv - Qcinv)@Q_vy).diagonal().sum()*self.r
            g_vz = 1/2*((Qinv - Qcinv)@Q_vz).diagonal().sum()*self.r
            g_noise = self.S.shape[0]*self.r/2 - 1/2*(Qcinv@self.S.transpose()@self.S*np.exp(par[7])).diagonal().sum()*self.r

            for j in range(self.r): # Maybe make a better version than this for loop possibly need to account for dimension 0
                g_kappa = g_kappa + (- 1/2*mu_c[:,j].transpose()@Q_kappa@mu_c[:,j])
                g_gammaX = g_gammaX + (- 1/2*mu_c[:,j].transpose()@Q_gammaX@mu_c[:,j])
                g_gammaX = g_gammaY + (- 1/2*mu_c[:,j].transpose()@Q_gammaY@mu_c[:,j])
                g_gammaY = g_gammaZ + (- 1/2*mu_c[:,j].transpose()@Q_gammaZ@mu_c[:,j])
                g_vx = g_vx + (- 1/2*mu_c[:,j].transpose()@Q_vx@mu_c[:,j])
                g_vy = g_vy + (- 1/2*mu_c[:,j].transpose()@Q_vy@mu_c[:,j])
                g_vz = g_vz + (- 1/2*mu_c[:,j].transpose()@Q_vz@mu_c[:,j])
                g_noise = g_noise + (- 1/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j] - self.S@mu_c[:,j])*np.exp(par[7]))
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[7])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = -like/(self.r * self.S.shape[0])
            jac = - np.array([g_kappa,g_gammaX,g_gammaY,g_gammaZ,g_vx,g_vy,g_vz,g_noise])/(self.r * self.S.shape[0])
            self.opt_steps = self.opt_steps + 1
            self.like = like
            self.jac = jac
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3_1 = %2.2f"%np.exp(par[1]),"\u03B3_2 = %2.2f"%np.exp(par[2]),"\u03B3_3 = %2.2f"%np.exp(par[3]),
                "\u03C1_1 = %2.2f"%(par[4]),"\u03C1_2 = %2.2f"%(par[5]),"\u03C1_3 = %2.2f"%(par[6]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[7])))
            #os.write(1, b'middle \n')
            return((like,jac))
        else: 
            like = 1/2*Q_fac.logdet()*self.r + self.S.shape[0]*par[7]*self.r/2 - 1/2*Q_c_fac.logdet()*self.r
            for j in range(self.r):
                like = like + (- 1/2*mu_c[:,j].transpose()@Q@mu_c[:,j] - np.exp(par[7])/2*(data[:,j] - self.S@mu_c[:,j]).transpose()@(data[:,j]-self.S@mu_c[:,j]))
            like = -like/(self.r * self.S.shape[0])
            self.like = like
            self.opt_steps = self.opt_steps + 1
            if self.verbose:
                print("# %4.0f"%self.opt_steps," log-likelihood = %4.4f"%(-like), "\u03BA = %2.2f"%np.exp(par[0]), "\u03B3_1 = %2.2f"%np.exp(par[1]),"\u03B3_2 = %2.2f"%np.exp(par[7]),"\u03B3_3 = %2.2f"%np.exp(par[3]),
                "\u03C1_1 = %2.2f"%(par[4]),"\u03C1_2 = %2.2f"%(par[5]),"\u03C1_3 = %2.2f"%(par[6]), "\u03C3 = %2.2f"%np.sqrt(1/np.exp(par[7])))
            return(like)

    