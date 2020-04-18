from .utils import *
from numba import jit
from numba import int16 as numba_int16
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
    ccf_regions = pd.DataFrame(ccf_regions)
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

########################################################################
########################################################################
########################################################################

def allen_proj_extent(proj, ccf_regions, foraxis=False):
    '''
Utility function to return the extent of normalized axis for when plotting 
allen projections on mm axes.

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
    
def allen_load_reference(reference_name,annotation_dir = annotation_dir):
    '''
Load allen areas to use as reference.

Example:
    ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')

    '''
    
    ccf_regions = pd.read_json(pjoin(
        annotation_dir,'{0}_ccf_labels.json'.format(reference_name)))
    proj = np.load(pjoin(annotation_dir,
                         '{0}_projection.npy'.format(reference_name)))
    brain_outline = np.load(pjoin(annotation_dir,
                                  '{0}_outline.npy'.format(reference_name)))
    return ccf_regions,proj,brain_outline

def allen_save_reference(ccf_regions, proj, brainoutline,
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

    
