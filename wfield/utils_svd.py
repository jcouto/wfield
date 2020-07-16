#  wfield - tools to analyse widefield data - svd utils 
# Copyright (C) 2020 Joao Couto - jpcouto@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from .utils import *

class svd_pix_correlation():
    def __init__(self,U,SVT,dims = None, norm_svt=False):
        '''
        Local correlation using the decomposed matrices.
        '''
        normed = SVT.copy()
        if issparse(U):
            if dims is None:
                raise ValueError('Supply dims when using with sparse arrays.')
            self.U = U
        else:
            self.dims = U.shape[:2]
            self.U = U.copy().reshape([-1,U.shape[-1]])
        if norm_svt:
            # Careful interpreting the results with normalized components...
            from sklearn.preprocessing import normalize
            normed = normalize(SVT,norm='l1',axis=1)
        # compute the covariance of the temporal components and estimate the pixelwise variance 
        self.cov_svt = np.cov(normed).astype('float32')
        self.var_pix = np.sum((self.U@self.cov_svt).T.conj()*self.U.T,axis = 0)
                
    def get(self,x,y):
        try:
            xy=np.ravel_multi_index([x,y],dims=self.dims)
        except ValueError:
            return None
        # Estimate the cov for a pixel from the covariance of the temporal components
        cov_pix = self.U[xy,:] @ self.cov_svt @ (self.U.T)
        # Normalize by the pixelwise
        std_pix = (self.var_pix[xy]**.5) * (self.var_pix**.5)
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
            fig.canvas.flush_events()
        fig.canvas.mpl_connect('button_press_event', update)
    
