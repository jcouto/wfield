#  wfield - tools to analyse widefield data - dask tools 
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

from .io import *
from dask import array as da
from dask import delayed
from dask.array import stats as dastats

def dask_load_binary(binfile,
                     nchan=None,
                     w=None,
                     h=None,
                     dtype=None,
                     blocksize = 512):
    assert os.path.isfile(binfile)
    meta = os.path.splitext(binfile)[0].split('_')
    if nchan is None: # try to get it from the filename
        nchan = int(meta[-4])
    if w is None: # try to get it from the filename
        w = int(meta[-3])
    if h is None: # try to get it from the filename
        h = int(meta[-2])
    if dtype is None: # try to get it from the filename
        dtype = meta[-1]
    dt = np.dtype(dtype)
    nframes = os.path.getsize(binfile)/(nchan*w*h*dt.itemsize)
    iblocks = np.arange(0,nframes,blocksize,dtype=int)
    if iblocks[-1] < nframes:
        iblocks = np.append(iblocks,nframes)
    framesize = int(nchan*w*h)
    blocks = []
    for ii,offset in enumerate(iblocks[:-1]):
        bsize = int(iblocks[ii+1] - offset)
        blocks.append((binfile,offset,bsize))
    lazyload = delayed(lambda x: load_binary_block(x,shape=(nchan,w,h)), 
                       pure=True)  # Lazy version of sbb_load_block
    sample = load_binary_block(blocks[0],shape=(nchan,w,h))
    lazy_values = [lazyload(block) for block in blocks]
    arrays = [da.from_delayed(lazy_value,           
                              dtype=sample.dtype,   
                              shape=[block[-1]] + [s for s in sample.shape[1:]])
          for block,lazy_value in zip(blocks,lazy_values)]
    return da.concatenate(arrays, axis=0)
