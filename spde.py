import sys, getopt
import numpy as np
import os
from scipy import sparse
from grid import Grid
import plotly.graph_objs as go
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
inla = importr("INLA")
#robj.r('inla.setOption("smtp" = "pardiso", pardiso.license = "~/OneDrive - NTNU/host_2020/pardiso.lic")')

def is_int(val):
    try:
        _ = int(val)
    except ValueError:
        return(False)
    return(True)

def rqinv(Q):
    tmp = Q.shape
    Q = Q.tocoo()
    r = Q.row
    c = Q.col
    v = Q.data
    tmpQinv = np.array(robj.r["as.data.frame"](robj.r["summary"](robj.r["inla.qinv"](robj.r["sparseMatrix"](i = robj.FloatVector(r+1),j = robj.FloatVector(c+1),x = robj.FloatVector(v))))))
    return(sparse.csc_matrix((np.array(tmpQinv[2,:],dtype = "float32"), (np.array(tmpQinv[0,:]-1,dtype="int32"), np.array(tmpQinv[1,:]-1,dtype="int32"))), shape=tmp))


class spde:
    '''
    Non-Stationary Anisotropic model in 2D and 3D. The Gaussian Random Field is approximated with a Stochastic Partial Differential Equation
    with the matérn covariance function as stationary solution. 
    

    Parameters
    ----------
    grid : class
        Self defined grid from the grid class. If not specified the grid will be default. 
    par : array
        Weights of the basis splines for each parameter. Must have length 136.
    model: int
        Defines the type of model (1 = stationary isotropic, 2 = stationary bidirectional anistropy, 
        3 = non-stationary unidirectional anisotropy, 4 = non-stationary biderectional anisotropy)

    Attributes
    ----------
    define : str
        This is where we store arg,
    '''
    def __init__(self, model = None):
        self.grid = Grid()
        self.model = model
        if self.model is not None:
            self.define(model = self.model)
        else:
            self.mod = None 

    def setGrid(self,M=None,N = None, P = None, x = None, y = None, z = None):
        self.grid.setGrid(M = M, N = N, P = P, x = x, y = y, z = z)
        if self.mod is not None:
            self.mod.setGrid(self.grid)

    # fix par for models
    def define(self, model = None, par = None):
        assert(model is not None or self.model is not None)
        if model is not None:
            self.model = model 
        if (self.model==1):
            from StatIso import StatIso
            self.mod = StatIso(grid = self.grid, par=par)
        elif (self.model==2):
            from StatAnIso import StatAnIso
            self.mod = StatAnIso(grid = self.grid, par=par)
        elif (self.model==3):
            from NonStatAnIso import NonStatAnIso
            self.mod = NonStatAnIso(grid = self.grid, par=par)
        elif (self.model==4):
            from NonStatAnIso3 import NonStatAnIso
            self.mod = NonStatAnIso(grid = self.grid,par=par)
        elif (self.model==5):
            from StatAnIso3 import StatAnIso
            self.mod = StatAnIso(grid = self.grid,par=par)
        else:
            print("Not a implemented model (1-4)...")

    def load(self,model=None,simple = False):
        if model is None:
            if self.mod is None:
                print("No model defined...")
            else:
                print("Loading pre-defined model",self.model)
        else:
            self.define(model = model)
        self.mod.load(simple = simple)

    def loadFit(self, simmod, dho, r, num, model = None, file = None):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.model = model
            self.define(model = model)
        self.mod.loadFit(simmod = simmod, dho = dho , r = r, num = num , file=file)

    # fix par for models
    def fitTo(self,simmod,dho,r,num,model = None,verbose = False, grad = True,par = None):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.model = model
            self.define(model = model)
        success = self.mod.fitTo(simmod, dho, r, num,verbose = verbose, grad=grad,par = par)
        return(success)
        
    # fix par for models
    def fit(self,data,r, S, par = None,model = None,verbose = False, grad = True):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.model = model
            self.define(model = model)
        res = self.mod.fit(data, r, S, par = par,verbose = verbose, grad=grad)
        return(res)

    def sim(self,model=None):
        if model is None:
            if self.mod is None:
                print("No model defined...")
            else:
                print("Loading pre-defined model ", self.model)
        else:
            self.define(model = model)
        return(self.mod.sim())

    #fix par for models
    def sample(self, n = 1,model = None, par = None):
        if model is None:
            if self.mod is None:
                print("No model defined...")
        else:
            self.define(model = model)
        return(self.mod.sample(n = n, par = par))

    def predict(self):
        return

    def update(self):
        return

    def Mvar(self):
        if self.mod.Q is None:
            self.mod.setQ()
        self.mod.mvar = rqinv(self.mod.Q).diagonal()

    # add to all

    def getPars(self):
        assert self.mod is not None
        return(self.mod.getPars())

    
    def Corr(self, pos = None):
        ik = np.zeros(self.mod.n)
        if pos is None:
            pos = np.array([self.grid.M/2,self.grid.N/2, self.grid.P/2])
        k = int(pos[1]*self.grid.M*self.grid.P + pos[0]*self.grid.P + pos[2])
        ik[k] = 1
        cov = self.mod.Q_fac.solve_A(ik)
        if self.mod.mvar is None:
            self.Mvar()
        return(cov/np.sqrt(self.mod.mvar[k]*self.mod.mvar))

    def plot(self,version="mvar", pos = None):
        if version == "mvar":
            if self.mod.mvar is None:
                self.Mvar()
            value = self.mod.mvar
        elif version == "mcorr":
            if self.mod.Q_fac is None:
                self.mod.setQ()
            if pos is None:
                pos = np.array([self.grid.M/2,self.grid.N/2, self.grid.P/2])
            value = self.Corr(pos)
        elif version == "real":
            value = self.sample()
        fig = go.Figure(go.Volume(
            x=self.grid.sx,
            y=self.grid.sy,
            z= - self.grid.sz,
            value = value,
            opacity=0.4, 
            surface_count=30,
            colorscale='rainbow',
            caps= dict(x_show=False, y_show=False, z_show=False)))
        fig.write_html("test.html",auto_open=True)


    
        



