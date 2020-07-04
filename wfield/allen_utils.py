#  wfield - tools to analyse widefield data - allen utils (numba) 
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

from numba import jit
from numba import int16 as numba_int16

@jit(nopython=True)
def allen_top_proj_from_volume(bvol):
    '''
Get the top projection from a volume.

    proj = allen_top_proj_from_volume(allen_volume)

'''
    h,d,w = bvol.shape
    proj = np.zeros((h,w),dtype=numba_int16)
    # this can be done with a np.where but is probably faster like this
    for i in range(h):
        for j in range(w):
            for z in range(d): 
                if bvol[i,z,j] > 0:
                    proj[i,j] = bvol[i,z,j]
                    break
            
    return proj
