from re import S
import numpy as np
import pandas as pd
from spde import spde
import netCDF4
import datetime
from scipy import sparse
from auv import auv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import norm
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.io as pio


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

class mission:
    def __init__(self,file):
        self.file = file
        self.emulator_exists = False
        self.assimilate_exists = False
        self.auv_exists = False
        self.muf = None
        self.mu = None

    def fit(self,par=None,verbose = False):
        assert self.emulator_exists, "No emulator defined"
        assert self.muf is not None, "No model mean defined"
        print("Starting model fit of model "+ str(self.mod.model))
        self.mod.fit(data=self.muf, par = par,r=self.muf.shape[1],S=self.S,verbose= verbose)
        par = self.mod.getPars()
        np.savez("./mission/" + self.file + '/model_' + str(self.mod.model)+ "_new" + '.npz', par = par)

    def init_auv(self):
        assert self.emulator_exists, "No emulator defined"
        assert self.assimilate_exists, "No assimilation defined"
        assert self.mu is not None, "No prior mean defined"
        self.auv = auv(self.mod,self.mu,self.mdata['sd'].mean())
        self.auv_exists = True

    def update(self, idx=None, fold=None, plot = False,keep = False):
        assert self.auv_exists, "No AUV defined"
        if idx is not None:
            self.auv.update(self.mdata['data'][idx],self.mdata['idx'][idx])
        elif fold is not None:
            if hasattr(fold, "__len__"):
                self.auv.update(self.mdata[np.array([x in fold for x in self.mdata['fold']])],keep = keep)
            else:
                self.auv.update(self.mdata[self.mdata['fold']==fold],keep = keep)
        else:
            print("No index for update defined")

    def predict(self,idx = None, fold = None):
        assert self.auv_exists, "No AUV defined"
        if idx is not None:
            return(self.auv.mu[self.mdata['idx'][idx]])
        elif fold is not None:
            if hasattr(fold, "__len__"):
                return(self.auv.mu[self.mdata['idx'][np.array([x in fold for x in self.mdata['fold']])]])
            else:
                return(self.auv.mu[self.mdata['idx'][self.mdata['fold']==fold]])
        else:
            return(self.auv.mu)

    def cross_validation(self,version = 1, fold = None, simple = False):
        assert self.auv_exists, "No AUV defined"
        if fold is None:
            if len(self.mdata['fold'].unique()) == 1:
                # place in different folds here, random
                pass
            else:
                fold = self.mdata['fold'].unique()
        if version == 1:
            err = np.zeros((len(fold),3))
            for i in range(len(fold)):
                self.update(fold = np.delete(fold,i))
                tmp = self.predict(fold = fold[i])
                sigma = np.sqrt(self.auv.mvar(simple = simple))
                err[i,0] = self.RMSE(tmp,self.mdata['data'][self.mdata['fold']==fold[i]])
                err[i,1] = self.CRPS(tmp,self.mdata['data'][self.mdata['fold']==fold[i]],sigma[self.mdata['idx'][self.mdata['fold']==fold[i]]])
                err[i,2] = self.logScore(tmp,self.mdata['data'][self.mdata['fold']==fold[i]],sigma[self.mdata['idx'][self.mdata['fold']==fold[i]]])
                self.auv.loadKeep()
            err_df = pd.DataFrame({'RMSE': err[:,0],'CRPS': err[:,1],'log-score': err[:,2]})
        return(err_df)

    def assimilate(self, idx = None, save = False, plot = False, processing = False, model = 4):
        if not self.emulator_exists:
            self.emulator(model = model)
        eta_hat_df = pd.read_csv("./mission/" + self.file + "/data/EstimatedState.csv")
        salinity_df  = pd.read_csv("./mission/" + self.file + "/data/Salinity.csv")
        salinity_df = salinity_df[salinity_df[" entity "] == " SmartX"]
        df = pd.merge_asof(eta_hat_df,salinity_df,on="timestamp",direction="nearest")
        circ = 40075000
        R = 6371 * 10**3
        rlat = (eta_hat_df[" lat (rad)"] + eta_hat_df[" x (m)"]*np.pi*2.0/circ).to_numpy()
        rlon = (eta_hat_df[" lon (rad)"] + eta_hat_df[" y (m)"]*np.pi*2.0/(circ*np.cos(rlat))).to_numpy()
        if idx is None:
            idx = np.arange(rlat.shape[0])
            tidx = np.zeros(idx.shape) + 1
        elif isinstance(idx, list):
            tidx = list()
            for i in range(len(idx)):
                for j in range(idx[i].shape[0]):
                    tidx.append(i+1)
            tidx = np.array(tidx)
            idx = np.hstack(idx)
        else: 
            tidx = np.zeros(idx.shape) + 1
        rlat = rlat[idx]*180/np.pi
        rlon = rlon[idx]*180/np.pi
        rsal = df[' value (psu)'].to_numpy()[idx]
        rz = df[' depth (m)'].to_numpy()[idx]
        timestamp = df['timestamp'].to_numpy()[idx]

        tmpx = R*(rlon-self.lon[0,0])*np.pi/180*np.cos((self.lat.min() + self.lat.max())/180*np.pi/2)
        tmpy = R*(rlat - self.lat[0,0])*np.pi/180
        rot = np.arctan((R*(self.lat[0,148] - self.lat[0,0])*np.pi/180)/(R*(self.lon[0,148]-self.lon[0,0])*np.pi/180*np.cos((self.lat.min() + self.lat.max())/180*np.pi/2)))
        rx = tmpx*np.cos(rot) + tmpy*np.sin(rot)
        ry = - tmpx*np.sin(rot) + tmpy*np.cos(rot)
        
        idxs = np.zeros(rx.size)
        for i in range(rx.size):
            idxs[i] = np.nanargmin((ry[i]-self.mod.grid.y)**2)*self.mod.grid.M*self.mod.grid.P +  np.nanargmin((rx[i]-self.mod.grid.x)**2)*self.mod.grid.P + np.nanargmin((rz[i]-self.mod.grid.z)**2)
        si = 0
        u_idx = list()
        u_data = list()
        u_sd = list()
        u_time = list()
        u_fold = list()
        for i in range(1,idxs.size):
            if idxs[i-1] != idxs[i]:
                ei = i
                u_idx.append(idxs[si])
                u_data.append(rsal[si:ei].mean())
                u_sd.append(rsal[si:ei].std())
                u_time.append(timestamp[si:ei].mean())
                u_fold.append(tidx[si])
                si = i
        self.mdata  = pd.DataFrame({'idx': np.array(u_idx).astype("int32"), 'data': u_data, 'sd': u_sd, 'timestamp': u_time, 'fold': u_fold})
        self.assimilate_exists = True
        if save:
            self.mdata.to_csv('./mission/' + self.file + '/data/assimilated.csv', index=False)
        if plot:
            self.plot_assimilate()

    def load_assimilate(self,model = 4,plot = False):
        if not self.emulator_exists:
            self.emulator(model = model)
        self.mdata = pd.read_csv('./mission/'+self.file+ '/data/assimilated.csv')
        self.assimilate_exists = True
        if plot:
            self.plot_assimilate()

    def plot_assimilate(self):
        assert self.assimilate_exists, "Non assimilation found"
        if self.mdata['fold'].max()>3:
            nrows = np.ceil(self.mdata['fold'].max()/3).astype('int32')
            ncols = 3
        else:
            nrows = self.mdata['fold'].max().astype('int32')
            ncols = 1
        im = mpimg.imread('./mission/'+self.file+'/AOOsmall.png')
        tmp = np.load('./mission/'+self.file+'/AOOdat.npy')
        im_lon = np.linspace(tmp[:,1].min(),tmp[:,1].max(),im.shape[1])
        im_lat = np.linspace(tmp[:,0].min(),tmp[:,0].max(),im.shape[0])
        fig, axs = plt.subplots(ncols = ncols,nrows= nrows,figsize =(ncols*7.5,nrows*7.5))
        for i in range(nrows):
            for j in range(ncols):
                if (i*3+j+1) > self.mdata['fold'].max().astype('int32'):
                    fig.delaxes(axs[i,j])
                tmp = self.mdata.loc[self.mdata['fold'] == (i*3+j+1)]
                pos_lat = self.slat[tmp['idx']]
                pos_lon = self.slon[tmp['idx']]
                y = np.zeros(pos_lat.shape)
                x = np.zeros(pos_lon.shape)
                for k in range(pos_lat.size):
                    y[k] = np.nanargmin((pos_lat[k]-im_lat)**2)
                    x[k] = np.nanargmin((pos_lon[k]-im_lon)**2)
                y = im_lat.shape[0] - y
                axs[i,j].imshow(im)
                axs[i,j].plot(x,y,'-ob',markersize=4)
                axs[i,j].set_axis_off()
                axs[i,j].text(100, 100, chr(i*3+j+97),fontweight="bold",fontsize=25)
        fig.tight_layout()
        plt.savefig('./mission/'+self.file+'/figures/folds.png', dpi=300)

    def check_emulator(self,model=4):
        assert self.emulator_exists, 'No emulator exists'
        assert self.assimilate_exists, 'No assimilation exists'
        time_emu = np.array([datetime.datetime.timestamp(x) for x in self.time])
        time_data = self.mdata['timestamp'].to_numpy()
        timeidx = np.zeros(time_data.shape)
        for i in range(time_data.shape[0]):
            timeidx[i] = np.nanargmin((time_emu-time_data[i])**2)
        timeidx = timeidx.astype('int32')
        err = (self.edata[self.mdata['idx'],timeidx]-self.mdata['data']).to_numpy()
        fig,ax = plt.subplots(figsize = (10,10))
        im = ax.scatter(self.szll[self.mdata['idx']],err, c = self.edata[self.mdata['idx'],timeidx],cmap = plt.get_cmap('coolwarm'))
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(0.5,10,step = 1),colors = 'black')
        ax.axhline(c = 'k')
        plt.savefig('./mission/'+ self.file + '/figures/SINMOD_err.png', dpi=300)

    def emulator(self,save=False, pars = False, model = 4, new = False):
        nc = netCDF4.Dataset('./mission/' + self.file + '/SINMOD.nc')
        tmp = np.load("./mission/" +self.file+"/mission.npz")
        par = None
        if pars:
            if new:
                par = np.load("./mission/" + self.file + "/model_" + str(model)+ "_new" + ".npz")['par']*1
            else:    
                par = np.load("./mission/" + self.file + "/model_" + str(model)+ ".npz")['par']*1
        self.mod = spde(model = model, par=par)
        xtmp = tmp['xdom']*1
        ytmp = tmp['ydom']*1
        ztmp = tmp['zdom']*1
        ttmp = tmp['tdom']*1
        xdom = np.arange(xtmp[0],xtmp[1],xtmp[2])
        ydom = np.arange(ytmp[0],ytmp[1],ytmp[2])
        zdom = np.arange(ztmp[0],ztmp[1],ztmp[2])
        tdom = np.arange(ttmp[0],ttmp[1],ttmp[2])
        ttmp = np.array(self.file.split('_')[1:],dtype="int32")
        self.time = [datetime.datetime(2000 + ttmp[2] ,ttmp[1],ttmp[0],0) + datetime.timedelta(minutes=x) for x in nc['time'][tdom]*24*60]

        x = np.array(nc['xc'][xdom])
        y = np.array(nc['yc'][ydom])
        z = np.array(nc['zc'][zdom])
        M = x.shape[0]
        N = y.shape[0]
        P = z.shape[0]

        self.mod.setGrid(M,N,P,x,y,z)

        data = np.array(nc['salinity'][tdom,zdom,ydom,xdom])
        data = data.swapaxes(0,3).swapaxes(1,2).swapaxes(0,1).reshape(z.shape[0]*x.shape[0]*y.shape[0],tdom.shape[0])
        S = sparse.diags((data[:,0]>0)*1)
        tmp = np.array(np.where(S.diagonal() != 0)).flatten()
        self.S = delete_rows_csr(S.tocsr(),np.where(S.diagonal() == 0))
        self.edata = self.S.transpose()@self.S@data
        self.mod.setQ(S = self.S)

        # Grid
        self.lon = np.array(nc['gridLons'][:,:])
        self.lat = np.array(nc['gridLats'][:,:])
        zll = np.array(nc['zc'][:])
        self.slon = np.zeros(M*N*P)
        self.slat = np.zeros(M*N*P)
        self.szll = np.zeros(M*N*P)
        t = 0
        for j in ydom:
            for i in xdom:
                for k in zdom:
                    self.slon[t] = self.lon[j,i]
                    self.slat[t] = self.lat[j,i]
                    self.szll[t] = zll[k]
                    t = t + 1
        self.emulator_exists = True
        if save:
            np.save('./mission/' + self.file + '/depth.npy', self.szll)
            np.save('./mission/' + self.file + '/lats.npy', self.slat)
            np.save('./mission/' + self.file + '/lons.npy', self.slon)
            #np.save('./mission/' + self.file + '/prior.npy', tm)
            tmp = np.zeros((4,4))
            tmp[:,0]=[x[0],x[x.shape[0]], x[x.shape[0]],x[0]]
            tmp[:,1]=[y[0],y[0], y[t.shape[0]],y[y.shape[0]]]
            tmp[:,2]=[self.slat[0],self.slat[(N-1)*M*P + 0*P + 0], self.slat[(0)*M*P + (M-1)*P + 0],self.slat[(N-1)*M*P + (M-1)*P + 0]]
            tmp[:,3]=[self.slon[0],self.slon[(N-1)*M*P + 0*P + 0], self.slon[(0)*M*P + (M-1)*P + 0],self.slon[(N-1)*M*P + (M-1)*P + 0]]
            np.save('./mission/' + self.file + '/grid.npy', tmp)
            self.mod.mod.setQ()
            Q = self.mod.mod.Q.tocoo()
            r = Q.row
            c = Q.col
            v = Q.data
            tmp = np.stack([r,c,v],axis=1)
            np.save('./mission/' + self.file + '/Q.npy',tmp)

    # have multiple types of mean (array)
    def mean(self,version1 = 'm',version2 = "mf", idx = None):
        if version1 == 'm':
            self.mu = self.edata.mean(axis = 1)
        elif version1 == "t":
            time_emu = np.array([datetime.datetime.timestamp(x) for x in self.time])
            time_data = self.mdata['timestamp'].to_numpy()
            idx = 0 if idx is None else idx
            idx = np.nanargmin((time_emu-time_data[idx])**2)
            self.mu = self.edata[:,idx]
        if version2 == "mf":
            data = self.S@self.edata
            self.muf = data - (data).mean(axis = 1)[:,np.newaxis]
        elif version2 == "tfc":
            data = self.S@self.edata
            mu = data.mean(axis = 1)
            rho = np.sum((data[:,1:]-mu[:,np.newaxis])*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]),axis = 1)/np.sum((data[:,:(data.shape[1]-1)] - mu[:,np.newaxis])**2,axis = 1)
            self.muf = (data[:,1:]-mu[:,np.newaxis]) - rho[:,np.newaxis]*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]) + np.random.normal(0,0.2,data.shape[0]*(data.shape[1]-1)).reshape(data.shape[0],data.shape[1]-1)
        elif version2 == "tf":
            data = self.S@self.edata
            mu = data.mean(axis = 1)
            rho = np.sum((data[:,1:]-mu[:,np.newaxis])*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]),axis = 1)/np.sum(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis])
            self.muf = (data[:,1:]-mu[:,np.newaxis]) - rho[:,np.newaxis]*(data[:,:(data.shape[1]-1)] - mu[:,np.newaxis]) + np.random.normal(0,0.2,data.shape[0]*(data.shape[1]-1)).reshape(data.shape[0],data.shape[1]-1)

    def logScore(self,pred,truth,sigma):
        return(- (norm.logpdf(truth,loc = pred,scale = sigma).mean()))

    def CRPS(self,pred,truth,sigma):
        z = (truth - pred)/sigma
        return(np.mean(sigma*(- 2/np.sqrt(np.pi) + 2*norm.pdf(z) + z*(2*norm.cdf(z)-1))))

    def RMSE(self,pred,truth):
        return(np.sqrt(np.mean((pred-truth)**2)))

    def plot(self,value = None, version = 1, filename = None):
        pio.orca.shutdown_server()
        M = self.mod.grid.M
        N = self.mod.grid.N
        P = self.mod.grid.P
        sx = self.mod.grid.sx
        sy = self.mod.grid.sy
        sz = self.mod.grid.sz
        if version == 1 or version == 2:
            cs = [(0.00, "rgb(127, 238, 240)"),   (0.50, "rgb(127, 238, 240)"),
                (0.50, "rgb(192, 245, 240)"), (0.60, "rgb(192, 245, 240)"),
                (0.60, "rgb(241, 241, 225)"),  (0.70, "rgb(241, 241, 225)"),
                (0.70, "rgb(255, 188, 188)"),  (0.80, "rgb(255, 188, 188)"),
                (0.80, "rgb(245, 111, 136)"),  (1.00, "rgb(245, 111, 136)")]
            if version == 1:
                if value is None:
                    assert self.mu is not None, "No mean assigned"
                    value = self.mu
                cmin = value.min()+2
                cmax = value.max()+2
                if filename is None:
                    filename = "mean"
            else: 
                if value is None:
                    self.mod.Mvar()
                    value = self.mod.mod.mvar
                cmin = value.min()
                cmax = value.max()
                if filename is None:
                    filename = "mvar"
        elif version == 3:
            cs = [(0.00, "rgb(245, 245, 245)"),   (0.20, "rgb(245, 245, 245)"),
                (0.20, "rgb(245, 201, 201)"), (0.40, "rgb(245, 201, 201)"),
                (0.40, "rgb(245, 164, 164)"),  (0.60, "rgb(245, 164, 164)"),
                (0.60, "rgb(245, 117, 117)"),  (0.80, "rgb(245, 117, 117)"),
                (0.80, "rgb(245, 67, 67)"),  (1.00, "rgb(245, 67, 67)")]
            if value is None:
                value = self.mod.Corr(pos = [22,2,0])
            cmin = 0
            cmax = value.max()
            if filename is None:
                filename = "mcorr"
        xarrow = np.array([sx.max()-175,sx.max()-50,sx.max()-50,sx.max()-50,sx.max()-125])
        yarrow = np.array([sy.max()-183,sy.max()-58,sy.max()-133,sy.max()-58,sy.max() -58])
        xdif = self.mod.grid.A/4
        ydif = self.mod.grid.B/4
        fig = go.Figure(data=[go.Isosurface(surface_count=1,z=-sz.reshape(M,N,P)[:,:,0].flatten(), x=sx.reshape(M,N,P)[:,:,0].flatten(), y=sy.reshape(M,N,P)[:,:,0].flatten(),value=value.reshape(M,N,P)[:,:,0].flatten(),isomin = cmin,isomax = cmax,colorscale=cs,colorbar=dict(thickness=20,lenmode = "fraction", len = 0.80, ticklen=10,tickfont=dict(size=30, color='black')))])
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(M,N,P)[:,:,1].flatten(), x=sx.reshape(M,N,P)[:,:,1].flatten()+xdif*1, y=sy.reshape(M,N,P)[:,:,1].flatten()-ydif*1,value=value.reshape(M,N,P)[:,:,1].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(M,N,P)[:,:,2].flatten(), x=sx.reshape(M,N,P)[:,:,2].flatten()+xdif*2, y=sy.reshape(M,N,P)[:,:,2].flatten()-ydif*2,value=value.reshape(M,N,P)[:,:,2].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(M,N,P)[:,:,3].flatten(), x=sx.reshape(M,N,P)[:,:,3].flatten()+xdif*3, y=sy.reshape(M,N,P)[:,:,3].flatten()-ydif*3,value=value.reshape(M,N,P)[:,:,3].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(M,N,P)[:,:,4].flatten(), x=sx.reshape(M,N,P)[:,:,4].flatten()+xdif*4, y=sy.reshape(M,N,P)[:,:,4].flatten()-ydif*4,value=value.reshape(M,N,P)[:,:,4].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))
        fig.add_trace(go.Isosurface(surface_count=1,z=-sz.reshape(M,N,P)[:,:,5].flatten(), x=sx.reshape(M,N,P)[:,:,5].flatten()+xdif*5, y=sy.reshape(M,N,P)[:,:,5].flatten()-ydif*5,value=value.reshape(M,N,P)[:,:,5].flatten(),isomin = cmin,isomax =cmax,colorscale=cs,showscale = False))

        fig.add_trace(go.Scatter3d(x=[0,0,xdif,xdif*2,xdif*3,xdif*4,xdif*5]+sx[0], y=[0,0,-ydif,-ydif*2,-ydif*3,-ydif*4,-ydif*5]+sy[0], z=[0,-0.5,-1.5,-2.5,-3.5,-4.5,-5.5], mode='text',text = ["Depth:","0.5","1.5","2.5","3.5","4.5","5.5"],textfont=dict(family="sans serif",size=25,color="black"),showlegend=False))
        for i in range(6):
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[0*45*10 + 0*10 + 0, 0*45*10 + 44*10 + 0]]+xdif*i, y=[0,0]+sy[[0*45*10 + 0*10 + 0, 0*45*10 + 44*10 + 0]]-ydif*i, z=np.array([-0.5,-0.5])-i, mode='lines',line = dict(color='black'),showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[0*45*10 + 0*10 + 0, 44*45*10 + 0*10 + 0]]+xdif*i, y=[0,0]+sy[[0*45*10 + 0*10 + 0, 44*45*10 + 0*10 + 0]]-ydif*i, z=np.array([-0.5,-0.5])-i, mode='lines',line = dict(color='black'),showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[44*45*10 + 0*10 + 0, 44*45*10 + 44*10 + 0]]+xdif*i, y=[0,0]+sy[[44*45*10 + 0*10 + 0, 44*45*10 + 44*10 + 0]]-ydif*i, z=np.array([-0.5,-0.5])-i, mode='lines',line = dict(color='black'),showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0,0]+sx[[0*45*10 + 44*10 + 0, 44*45*10 + 44*10 + 0]]+xdif*i, y=[0,0]+sy[[0*45*10 + 44*10 + 0, 44*45*10 + 44*10 + 0]]-ydif*i, z=np.array([-0.5,-0.5])-i, mode='lines',line = dict(color='black'),showlegend=False))


        fig.add_trace(go.Scatter3d(x=xarrow, 
                                        y=yarrow,
                                        z=np.array([0,0,0,0,0])-0.5,
                                        line=dict(color='black',width=12),
                                        mode='lines+text',
                                        text=["","", "N","",""],
                                        showlegend=False,
                                        textfont=dict(
                                            family="sans serif",
                                            size=25,
                                            color="LightSeaGreen")))

        camera = dict(
                    eye=dict(x=1.2, 
                            y=-1.2, 
                            z=1.3),
                    center=dict(x=0.2, y=-0.2, z=0.18))
        fig.update_scenes(xaxis_visible=False, 
                            yaxis_visible=False,zaxis_visible=False,camera = camera)
        fig.update_layout(autosize=False,
                        width=650,height=1200,scene_aspectratio=dict(x=1, y=1, z=1.0))
        fig.write_html("test.html", auto_open = True)
        #fig.write_image("./mission/" + self.file + "/figures/" + filename + ".png",engine="orca",scale=1)