from .utils import *
from sklearn.preprocessing import normalize

class svd_pix_correlation():
    def __init__(self,U,SVT,dims,norm_svt=False):
        '''
        Local correlation using the decomposed matrices.
        '''
        normed = SVT.copy()
        self.U = U
        self.dims = dims
        if norm_svt:
            # Careful interpreting the results with normalized components...
            normed = normalize(SVT,norm='l2',axis=0)
        # compute the covariance of the temporal components and estimate the pixelwise variance 
        self.cov_svt = np.cov(normed).astype('float32')
        self.var_pix = np.sum(np.conj((self.U @ self.cov_svt).T)*self.U.T, axis = 0)
        
    def get(self,x,y):
        xy=np.ravel_multi_index([x,y],dims=self.dims)
        # Estimate the cov for a pixel from the covariance of the temporal components
        cov_pix = ((self.U[xy,:] @ self.cov_svt) @ self.U.T)
        # Normalize by the pixelwise
        std_pix = (self.var_pix[xy]**.5) * self.var_pix**.5
        cov_pix[xy] = np.nan
        return (cov_pix/std_pix).reshape(self.dims)

    def pylab_show(self):
        import pylab as plt
        fig = plt.figure()
        ax = fig.add_axes([.1,.1,.8,.8])
        self.pylab_im = ax.imshow(self.get(0,0),clim=[-1,1],cmap='RdBu_r')

        def update(event):
            if event.inaxes == ax:
                xy = [int(event.ydata),int(event.xdata)]
                self.pylab_im.set_data(self.get(*xy))
                fig.canvas.draw()
            fig.cancas.flush_events()
        fig.canvas.mpl_connect('button_press_event', update)
    
