#  wfield - tools to analyse widefield data - allen tools 
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
from skimage import measure
from skimage.filters import gaussian
from scipy import ndimage
import json

# allensdk imports are inside the functions because it needs an old pandas and i dont understand why. To install it do: pip install allensdk

annotation_dir = pjoin(os.path.expanduser('~'),'.wfield')

selection_dorsal_cortex = [
    'MOB',
    'FRP',
    'MOp',
    'MOs',
    'SSp-n',
    'SSp-m',
    'SSp-un',
    'PL',
    'ACAd',
    'RSPv',
    'RSPd',
    'RSPagl',
    'VISC',
    'SSs',
    'SSp-bfd',
    'SSp-tr',
    'SSp-ll',
    'SSp-ul',
    'TEa',    
    'AUDd',
    'AUDp',
    'AUDpo',
    'AUDv',
    'VISli',
    'VISpor',
    'VISpl',
    'VISpm',
    'VISl',
    'VISal',
    'VISrl',
    'VISa',
    'VISam',
    'VISp',]

selection_vis_som = ['RSPv',
                     'RSPd',
                     'RSPagl',
                     'SSs',
                     'SSp-bfd',
                     'SSp-tr',
                     'SSp-ll',
                     'SSp-ul',
                     'SSp-un',
                     'SSp-n',
                     'VISli',
                     'VISpor',
                     'VISp',
                     'VISpm',
                     'VISl',
                     'VISal',
                     'VISrl',
                     'VISpl',
                     'VISa',
                     'VISam']

def allen_get_raw_annotation(annotation_dir,
                             version = 'annotation/ccf_2017',
                             resolution = 10):
    import nrrd
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
    
    annotation_path = pjoin(annotation_dir, 'annotation_{0}_{1}.nrrd'.format(
        version,resolution))
    if not os.path.isdir(annotation_dir):
        os.makedirs(annotation_dir)
    if not os.path.isfile(annotation_path):
        mcapi = MouseConnectivityApi()
        mcapi.download_annotation_volume(version, resolution, annotation_path)
    annotation, meta = nrrd.read(annotation_path)
    return annotation,meta

def allen_volume_from_structures(
        structures = selection_dorsal_cortex,
        resolution=10,
        version = 'annotation/ccf_2017'):
    '''
    
    Gets specific regions from an annotation volume from the allen atlas

    '''
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
    from allensdk.api.queries.ontologies_api import OntologiesApi
    from allensdk.core.structure_tree import StructureTree
    from allensdk.core.reference_space import ReferenceSpace
    # the annotation download writes a file, so we will need somwhere to put it

    # the annotation download writes a file, so we will need somwhere to put it
    annotation, meta = allen_get_raw_annotation(
        annotation_dir,
        version = version,
        resolution = 10)
    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])  # 1 is the id of the adult mouse structure graph

    # This removes some unused fields returned by the query
    structure_graph = StructureTree.clean_structures(structure_graph)  
    tree = StructureTree(structure_graph)    
    rsp = ReferenceSpace(tree, annotation, [resolution]*3)

    areas = rsp.structure_tree.get_structures_by_acronym(structures)
    ids = [st['id'] for st in areas]
    mask_volume = np.zeros_like(annotation,dtype='int16')
    for i,sid in tqdm(enumerate(ids)):
        masks = rsp.make_structure_mask([sid])
        mask_volume[masks == 1] = i+1
        areas[i]['mask_volume_id'] = i+1
    return  mask_volume, areas
    
def allen_flatten_areas(areas,mask_volume,
                        resolution=10, reference= [540,570],gaussfilt=0):    
    ''' 
Creates a top view and extracts contours from each area in an annotation volume 
    proj, ccf_regions = allen_flatten_areas(areas, mask_volume, 
                                            resolution=10, 
                                            reference= [540 570])
    
    The default reference point is a (certainly wrong) guess of the location of bregma.

'''
    from .allen_utils import allen_top_proj_from_volume
    proj = allen_top_proj_from_volume(mask_volume)
    uproj = np.unique(proj)
    h,w = proj.shape
    proja = proj.copy()
    proja[:,-int(w/2):] = 0
    mask_ids = [a['mask_volume_id'] for a in areas]
    ccf_regions = []
    res = resolution/1000.
    for i,v in enumerate(mask_ids):
        # find the largest region for each brain area
        a = (proja==v)
        if not gaussfilt==0:
            a = gaussian(a,sigma=gaussfilt)
        a = measure.label(a)
        rprops = measure.regionprops(a)
        if not len(rprops):
            print('No projection for '+areas[i]['acronym'])
            continue
        ar = np.argmax([r.area for r in rprops])
        c = measure.find_contours(a==rprops[ar].label,.5)[0]
        cm = np.array(rprops[ar].centroid)
        left_center = cm-np.array(reference)
        d = c.copy()
        d[:,1] = -1*(c[:,1])+w
        right_center = [1,-1]*cm+w-np.array(reference)

        ccf_regions.append(dict(acronym=areas[i]['acronym'],
                                name=areas[i]['name'],
                                reference = reference,
                                resolution = resolution,
                                label = v,
                                allen_id = areas[i]['id'],
                                allen_rgb = areas[i]['rgb_triplet'],
                                left_center = left_center[::-1]*res,
                                right_center = right_center[::-1]*res,
                                left_x = (c[:,1] - reference[1])*res,
                                left_y = (c[:,0] - reference[0])*res,
                                right_x = (d[:,1] - reference[1])*res,
                                right_y = (d[:,0] - reference[0])*res))
    from pandas import DataFrame
    ccf_regions = DataFrame(ccf_regions)
    return proj,ccf_regions

def projection_outline(proj, resolution = 10, reference=[540,570]):
    '''
Get the outline from the projection

    outline_contour = projection_outline(proj, resolution = 0.010)
    
    '''
    
    a = ndimage.binary_closing(proj >0,np.ones((10,10)))
    a = ndimage.binary_fill_holes(a)
    a = ndimage.binary_dilation(a)
    return (measure.find_contours(a,.5)[0][:,::-1]- np.array(reference)[::-1])*resolution/1000. 



########################################################################
########################################################################
########################################################################

def allen_proj_extent(proj, ccf_regions, foraxis=False):
    '''
Utility function to return the extent of normalized axis for when plotting 
allen projections on matplotlib axes.

    extent = allen_proj_extent(proj, ccf_regions, foraxis=False)
    
    proj is the projection (used to get dims)
    ccf_refions is a dataframe of regions (to get the reference and resolution)
    foraxis = True flips the Y axis
    
Example:
    plt.imshow(proj,extent = allen_proj_extent(proj,ccf_regions),
               origin='top', cmap = 'gray')
    plt.plot(brainoutline[:,0],brainoutline[:,1],'c',lw=3)    
    plt.axis(allen_proj_extent(proj,ccf_regions, foraxis=True))

    '''
    h,w = proj.shape
    r2,r1 = ccf_regions.reference.iloc[0]
    extent = np.array([-r1,w-r1,-r2,h-r2])*ccf_regions.resolution.iloc[0]/1000.
    if foraxis:
        return extent[[0,1,3,2]]
    else:
        return extent

########################################################################
########################################################################
########################################################################


def apply_affine_to_points(x,y,M):
    '''
    Apply an affine transform to a set of contours or (x,y) points.

    x,y = apply_affine_to_points(x, y, tranform)

    '''
    if M is None:
        nM = np.identity(3,dtype = np.float32)
    else:
        nM = M.params
    xy = np.vstack([x,y,np.ones_like(y)])
    res = (nM @ xy).T
    return res[:,0],res[:,1]

def allen_transform_regions(M,ccf_regions,resolution = 1,bregma_offset = [0.,0.]):
    ''' This transforms regions from the reference to image coordinates.
    Usage:

    lmarks = load_allen_landmarks('dorsal_cortex_landmarks.json')
    refregions = allen_load_reference('dorsal_cortex')
    refregions_image = allen_transform_regions(lmarks['transform'],refregions,
                                        resolution = lmarks['resolution'],
                                        bregma_offset = lmarks['bregma_offset'])

    The output: refregions_image is the same as refregions but in image space.
 '''
    nccf = ccf_regions.copy()
    for i,c in nccf.iterrows():
        for side in ['left','right']:
            x,y = apply_affine_to_points(np.array(c[side+'_x'])/resolution + bregma_offset[0],
                                         np.array(c[side+'_y'])/resolution + bregma_offset[1], M)
            nccf.at[i,side+'_x'] = x.tolist()
            nccf.at[i,side+'_y'] = y.tolist()
            x,y = apply_affine_to_points(c[side+'_center'][0]/resolution + bregma_offset[0],
                                         c[side+'_center'][1]/resolution + bregma_offset[1], M)
            nccf.at[i,side+'_center'] = [x[0],y[0]]
    return nccf

def allen_transform_from_landmarks(landmarks_im,match):
    '''
    Compute the similarity transform from annotated landmarks. 
    
    transform = allen_transform_from_landmarks(landmarks_im,match)
    
    '''
    ref = np.vstack([landmarks_im['x'],landmarks_im['y']]).T
    cor = np.vstack([match['x'],match['y']]).T
    return estimate_similarity_transform(ref, cor)

def allen_landmarks_to_image_space(landmarks,
                                   bregma_offset = np.array([0,0]),
                                   resolution = 0.01):
    '''
    Convert landmarks from allen to "image" space.
    Basically just divides by the resolution and adds the bregma offset.

    Warning this does the operation in place, pass .copy()

    landmarks = allen_landmarks_to_image_space(landmarks.copy(),
                                   bregma_offset = np.array([0,0]),
                                   resolution = 0.01)
    '''
    landmarks['x'] = landmarks['x']/resolution + bregma_offset[0]
    landmarks['y'] = landmarks['y']/resolution + bregma_offset[1]
    return landmarks


#######################################################################
################      READ AND WRITE FUNCTIONS      ###################
#######################################################################

def save_allen_landmarks(landmarks,
                         filename = None,
                         resolution = None,
                         landmarks_match = None,
                         bregma_offset = None,
                         transform = None,
                         **kwargs):
    '''
    landmarks need to be pandas dataframes.

    default is dorsal_cortex_landmarks.json in the ~/.wfield directory.

    '''
    lmarks = dict(landmarks=landmarks.to_dict(orient='list'))
    if not resolution is None:
        lmarks['resolution'] = resolution    
    if not landmarks_match is None:
        lmarks['landmarks_match'] = landmarks_match.to_dict(orient='list')
    if not bregma_offset is None:
        if isinstance(bregma_offset,np.ndarray):
            bregma_offset = bregma_offset.tolist()
        lmarks['bregma_offset'] = bregma_offset
    if not transform is None:
        from skimage.transform import SimilarityTransform
        if isinstance(transform,SimilarityTransform):
            lmarks['transform'] = transform.params.tolist()
        elif isinstance(transform,np.ndarray):
            lmarks['transform'] = transform.tolist()
        else:
            lmarks['transform'] = transform
    if 'bregma_offset' in lmarks.keys() and 'resolution' in lmarks.keys():
        lmarks['landmarks_im'] = allen_landmarks_to_image_space(
            landmarks.copy(), 
            lmarks['bregma_offset'],
            lmarks['resolution']).to_dict(orient='list')
    if filename is None:
        filename = pjoin(annotation_dir,'dorsal_cortex_landmarks.json')
    with open(filename,'w') as fd:
        import json
        json.dump(lmarks,fd, sort_keys = True, indent = 4)

def load_allen_landmarks(filename, reference = 'dorsal_cortex'):
    if filename is None:
        filename = pjoin(annotation_dir,reference + '_landmarks.json')
    with open(filename,'r') as fd:
        import json
        lmarks = json.load(fd)
    for k in ['landmarks_im','landmarks','landmarks_match']:
        if k in lmarks.keys():
            from pandas import DataFrame
            lmarks[k] = DataFrame(lmarks[k])[['x','y','name','color']]
    if 'transform' in lmarks.keys():
        from skimage.transform import SimilarityTransform
        lmarks['transform'] = SimilarityTransform(
            np.array(lmarks['transform']))
    return lmarks

    
def allen_load_reference(reference_name,annotation_dir = annotation_dir):
    '''
Load allen areas to use as reference.

Example:
    ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')

    '''
    from pandas import read_json
    ccf_regions = read_json(pjoin(
        annotation_dir,'{0}_ccf_labels.json'.format(reference_name)))
    proj = np.load(pjoin(annotation_dir,
                         '{0}_projection.npy'.format(reference_name)))
    brain_outline = np.load(pjoin(annotation_dir,
                                  '{0}_outline.npy'.format(reference_name)))
    return ccf_regions,proj,brain_outline

def allen_save_reference(ccf_regions,
                         proj,
                         brainoutline,
                         reference_name,
                         annotation_dir=annotation_dir):
    '''
    Save Allen references to the default directory

    allen_save_reference(ccf_regions, proj, brainoutline,
                         referece_name,
                         annotation_dir=annotation_dir):
    '''
    
    ccf_regions.to_json(pjoin(
        annotation_dir,'{0}_ccf_labels.json'.format(reference_name)),
                        orient='records')
    np.save(pjoin(annotation_dir,
                  '{0}_projection.npy'.format(reference_name)),proj)
    np.save(pjoin(annotation_dir,
                  '{0}_outline.npy'.format(reference_name)),brainoutline)

    

def allen_regions_to_atlas(ccf_regions,dims,
                           sides = ['left','right'],
                           fillnan = False):
    ''' 
    Atlas as called in locaNMF; it is the masks of allen areas.
    This function returns also the names of the areas
    '''
    atlas = np.zeros(dims,dtype = np.float32)
    if fillnan:
        atlas.fill(np.nan)
    areanames = []
    for ireg,r in ccf_regions.iterrows():
        for iside,side in enumerate(sides):
            mask = contour_to_mask(
                r[side+'_x'],r[side+'_y'],
                dims = dims)
            factor = 1
            if iside==1:
                factor = -1
            atlas[mask==1] = factor*(ireg+1)
            areanames.append([factor*(ireg+1),r['acronym']+'_'+side])
    return atlas,areanames

def atlas_from_landmarks_file(landmarks_file=None, reference='dorsal_cortex', dims = [540,640], do_transform = None):
    lmarks = load_allen_landmarks(landmarks_file)
    ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')
    # transform the regions into the image
    if not 'transform' in lmarks.keys():
        lmarks['transform'] = None
    transform = lmarks['transform']
    if not do_transform:
        transform = None
    nccf_regions = allen_transform_regions(transform,
                                           ccf_regions,
                                           resolution=lmarks['resolution'],
                                           bregma_offset=lmarks['bregma_offset'])
    nbrain_outline = apply_affine_to_points(brain_outline[:,0]/lmarks['resolution'] + lmarks['bregma_offset'][0],
                                            brain_outline[:,1]/lmarks['resolution'] + lmarks['bregma_offset'][1],
                                            transform)


    atlas,areanames = allen_regions_to_atlas(nccf_regions, dims)
    brain_mask = contour_to_mask(*nbrain_outline, dims = dims)
    return atlas, areanames, brain_mask

########################################################################
################HOLOVIEWS PLOTTING FUNCTIONS############################
########################################################################

def hv_plot_allen_regions(ccf_regions,
                          resolution = 1,
                          bregma_offset = np.array([0,0]),
                          side_selection='both'):
    '''
    Example: 

import holoviews as hv
hv.extension('bokeh')

ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')
hv_plot_allen_regions(ccf_regions).options({'Curve': {'color':'black', 'width': 600}})
    
    '''
    import holoviews as hv
    regs = []
    for p in ccf_regions.iterrows():
        c = p[1]
        if side_selection in ['right','both']:
            regs.append(hv.Curve(np.vstack([np.array(c.right_x)/resolution + bregma_offset[0],
                                            np.array(c.right_y)/resolution + bregma_offset[1]]).T))
        if side_selection in ['left','both']:
            regs.append(hv.Curve(np.vstack([np.array(c.left_x)/resolution + bregma_offset[0],
                                            np.array(c.left_y)/resolution + bregma_offset[1]]).T))
    # return a plot
    plot = regs[0]
    for i in range(1,len(regs)):
        plot *= regs[i]
    return plot.opts(invert_yaxis=True).options(width = 600)

def hv_adjust_reference_landmarks(landmarks,ccf_regions,msize=40):
    '''
    landmarks = {'x': [-1.95, 0, 1.95, 0],
                 'y': [-3.45, -3.45, -3.45, 3.2],
                 'name': ['OB_left', 'OB_center', 'OB_right', 'RSP_base'],
                 'color': ['#fc9d03', '#0367fc', '#fc9d03', '#fc4103']}
    landmarks = pd.DataFrame(landmarks)
    # adjust landmarks
    wid,landmark_wid = hv_adjust_reference_landmarks(landmarks,ccf_regions)
    wid # to display
    # use the following to retrieve (on another cell) 
    landmarks = pd.DataFrame(landmark_wid.data)[['x','y','name','color']]
    '''
    import holoviews as hv
    from holoviews import opts, streams
    from holoviews.plotting.links import DataLink
    referenceplt = hv_plot_allen_regions(ccf_regions).options(
        {'Curve': {'color':'black', 'width': 600}})

    points = hv.Points(landmarks,vdims='color').opts(marker='+',size=msize)
    point_stream = streams.PointDraw(data=points.columns(), 
                                     add = False,num_objects=4, 
                                     source=points, empty_value='black')
    table = hv.Table(points, ['x', 'y','name'], 'color').opts(title='Landmarks location')
    DataLink(points, table)
    widget = (referenceplt*points + table).opts(
        opts.Layout(merge_tools=False),
        opts.Points(invert_yaxis=True,
                    active_tools=['point_draw'],
                    color='color', height=500,
                    tools=['hover'], width=500),
        opts.Table(editable=True))
    return widget,point_stream

def hv_adjust_image_landmarks(image,landmarks,
                              landmarks_match = None,
                              bregma_offset = None,
                              resolution = 0.0194,
                              msize = 40):
    '''
    TODO: merge part of this with the one for the landmarks
    landmarks are in allen reference space
    '''
    h,w = image.shape
    if bregma_offset is None:
        # then it is the center of the image
        bregma_offset = np.array([int(w/2),int(h/2)]) # place bregma in the center of the image
        
    landmarks_im = allen_landmarks_to_image_space(landmarks.copy(),bregma_offset,resolution)
    if landmarks_match is None:
        landmarks_match = landmarks_im
    import holoviews as hv
    from holoviews import opts, streams
    from holoviews.plotting.links import DataLink

    bounds = np.array([0,0,w,h])
    im = hv.Image(image[::-1,:],
                 bounds =tuple(bounds.tolist())).opts(
        invert_yaxis = True,cmap = 'gray')

    points = hv.Points(landmarks_match,vdims='color').opts(marker='+',size=msize)
    point_stream = streams.PointDraw(data=points.columns(), 
                                     add = False,num_objects=4, 
                                     source=points, empty_value='black')
    table = hv.Table(points, ['x', 'y','name'], 'color').opts(title='Annotation location')
    DataLink(points, table)

    from bokeh.models import HoverTool
    hoverpts = HoverTool(tooltips=[("i", "$index")])

    widget = (im*points + table).opts(
        opts.Layout(merge_tools=False),
        opts.Points(invert_yaxis=True,active_tools=['point_draw'], 
                    color='color',
                    tools=[hoverpts], 
                    width=int(w),
                    height=int(h)),
        opts.Table(editable=True))
    return widget,point_stream,landmarks_im

def hv_show_transformed_overlay(image, M, ccf_regions,
                                bregma_offset = None,resolution=0.0194):
    import holoviews as hv
    h,w = image.shape
    if bregma_offset is None:
        bregma_offset = np.array([int(w/2),int(h/2)]) # place bregma in the center of the image
    bounds = np.array([0,0,w,h])
    warped = warp(image,M,
                  order = 0,
                  clip=True,
                  preserve_range=True)
    im=hv.Image(warped[::-1,:],
                 bounds =tuple(bounds.tolist())).opts(invert_yaxis = True,cmap = 'gray')
    return im*hv_plot_allen_regions(ccf_regions,
                                    bregma_offset=bregma_offset,
                                    resolution=resolution).options(
        {'Curve': {'color':'white', 'width': w,'height':h}})


