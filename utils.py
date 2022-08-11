import os
from time import time
import tempfile

import numpy as np 
import pyvista as pv
import pyacvd
import SimpleITK as sitk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import vtk



def write_as_file(file_kwargs,):
    """
    """
    for path, var in file_kwargs.items():
        if path[-len('.vtk'):] == '.vtk': # write as vtk file
            pv.wrap(var).save(path) 

        elif path[-len('.mha'):] == '.mha': # write with SimpleITK.WriteImage
            sitk.WriteImage(var, path, True)

        else: 
            raise NotImplententedError(f'Cannot write {path}.')


def stopwatch_toggle(nm, trace=None):
    """ 
    """
    if trace is None: 
        trace = {}
    if nm in trace: 
        trace[nm] = time() - trace[nm] 
    else:
        trace[nm] = time()
    return trace 


def load_file(*file_args,):
    """
    """
    returns = []
    for path in file_args:
        if path[-len('.vtk'):] == '.vtk': # write as vtk file
            returns.append(pv.read(path))

        elif path[-len('.mha'):] == '.mha': # write with SimpleITK.WriteImage
            returns.append(sitk.ReadImage(path))

        else: 
            raise NotImplententedError(f'Cannot read {path}.')
    return tuple(returns)


def np_to_im(np_im, ref_im, pixel_type):
    """Save numpy array as itk image (volume) taking im parameters (spacing, origin, direction) from ref_im"""
    im_out = sitk.GetImageFromArray(np_im)
    im_out = sitk.Cast(im_out, pixel_type)
    im_out.SetSpacing(ref_im.GetSpacing())
    im_out.SetOrigin(ref_im.GetOrigin())
    im_out.SetDirection(ref_im.GetDirection())
    return im_out


def get_center_of_mass(m):
    """ Get center of mass of mesh m as numpy array"""
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(m)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = np.array(centerOfMassFilter.GetCenter())
    return center


def pointthreshold(polydata, arrayname, start=0, end=1, alloff=0):
    threshold = vtk.vtkThreshold()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        threshold.SetInputData(polydata)
    else:
        threshold.SetInput(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)
    threshold.ThresholdBetween(start, end)
    if (alloff):
        threshold.AllScalarsOff()
    threshold.Update()

    surfer = vtk.vtkDataSetSurfaceFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfer.SetInputData(threshold.GetOutput())
    else:
        surfer.SetInput(threshold.GetOutput())
    surfer.Update()
    return surfer.GetOutput()


def extractlargestregion(polydata):
    """Keep only biggest region"""
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(polydata)
    surfer.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surfer.GetOutput())
    cleaner.Update()

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(cleaner.GetOutput())
    connect.SetExtractionModeToLargestRegion()
    connect.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(connect.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()  

def path_triagulate(mesh):
    """ Transform 2-vertice face to 3-vertice, by duplicate the index.
    """
    mesh = pv.wrap(mesh)
    faces = mesh.faces
    _face = faces.reshape([-1, faces[0]+1])[:, 1:]
    _face = np.hstack([3*np.ones(len(_face))[..., None], _face, _face[:, -1][..., None]])
    mesh.faces = _face.astype(int)
    return mesh

def check_360(thetas, tolerance=0.10):
    output = True
    ref_array = np.arange(-np.pi, np.pi, 2*np.pi/360)
    for i in range(len(ref_array)):
        # find closest value in thetas
        idx = (np.abs(thetas - ref_array[i])).argmin()
        val = thetas[idx]
        if np.abs(val-ref_array[i]) > tolerance:
            output = False
    return output


def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta    


def read_mhaimage(filename):
    """Read .mha file"""
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def probe_img_2_mesh(mesh, 
        img):
    """ Probe img value to mesh
    """
    # write image as temp file
    tmp_path = 'probing_tmp.mha'
    sitk.WriteImage(img, tmp_path)
    
    meta_img = read_mhaimage(tmp_path)

    probe = vtk.vtkProbeFilter()
    probe.SetSourceData(meta_img)
    probe.SetInputData(mesh)
    probe.Update()
    output_mesh = probe.GetOutput()
    
    os.remove(tmp_path)
    return pv.wrap(output_mesh)


def uniform_meshing(polydata, n_clusters=1000, subdivide=False,):
    """
    """
    wrapped_data = pv.wrap(polydata)
    clus = pyacvd.Clustering(wrapped_data)
    if subdivide:
        clus.subdivide(3)
    clus.cluster(n_clusters)
    polydata = clus.create_mesh()
    return polydata


def clean_polydata(polydata, tolerance=0):
    """
    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetDebug(1)
    cleaner.SetTolerance(tolerance)
    cleaner.SetInputData(polydata)
    cleaner.Update()
    polydata = cleaner.GetOutput()
    return polydata

def BinaryMorphologicalClosing(img, kernel_radius,):
    """
    """
    filt = sitk.BinaryMorphologicalClosingImageFilter()
    filt.SetKernelRadius(kernel_radius)
    return filt.Execute(img)

def marching_cubes(
    img,
    decimate=0.7,
    smoothing_iterations=30,
    relaxation_factor=0.2,
    closing_radius=0,
    tmp_file=None,
    rm_tmpfile=True,
    n_clusters=10000,
    tolerance=0,
    clean=True,
    # tetra=False,
):
    """
    Creates a mesh from a binary mask.

    Quite a mess right now, because returns a volumetric pyvista mesh if tetra=True
    and a surface cardiac_utils.mesh.SpatialData otherwise.
    Still useful because VTK ignores the "direction" information from imaging
    data.

    Maybe someday I'll find the courage to clean that without breaking everything
    else.

    :param decimate: VTK marching cubes parameter
    :param smoothing_iterations: VTK marching cubes parameter
    :param relaxation_factor: VTK marching cubes parameter
    :param closing_radius: Apply binary closing before meshing
    :param tmp_file:
    :param rm_tmpfile:
    :param n_clusters: Number of points in the mesh
    :param clean: VTK marching cubes parameter
    :param tetra: True for a volumetric, False for a surface mesh
    :return:
    """
    mask = img

    if closing_radius:
        # mask = closing(mask, selem=ball(closing_radius))
        mask = BinaryMorphologicalClosing(mask, closing_radius)

    if tmp_file is None:
        handle, tmp_file = tempfile.mkstemp(suffix=".mha")
        os.close(handle)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    sitk.WriteImage(mask, tmp_file)

    vtkimg = read_mhaimage(tmp_file)

    # apply marching cubes
    triangulator = vtk.vtkDiscreteMarchingCubes()
    triangulator.SetInputData(vtkimg)
    triangulator.Update()
    polydata = triangulator.GetOutput()

    if decimate:
        # decimation
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(polydata)
        decimator.SetTargetReduction(decimate)
        decimator.Update()
        polydata = decimator.GetOutput()

    if smoothing_iterations:
        # smoothing
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.SetInputData(polydata)
        smoother.Update()
        polydata = smoother.GetOutput()

    if n_clusters is not None:
        polydata = uniform_meshing(polydata, n_clusters=n_clusters)

    if clean:
        polydata = clean_polydata(polydata, tolerance=tolerance)

    if rm_tmpfile:
        os.remove(tmp_file)

    return pv.wrap(polydata)
