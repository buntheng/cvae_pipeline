import os
import tempfile
import math
import pdb

import SimpleITK as sitk
import numpy as np
import pyvista as pv
import pyacvd 
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import utils


def read_sitk(path:str): 
    """
    """
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        filenames = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(filenames)
        img = reader.Execute()
    else: 
        img = sitk.ReadImage(path)
    return img


def get_rotation_transform(R, 
        img,
        reference_image,
        reference_origin,
        reference_center,
        default_pixel_value=0,
        return_direction=False,
        ):
    """
    """
    dimension = img.GetDimension()
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    
    # centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)
    
    rotate_transform = sitk.AffineTransform(dimension)
    rotate_transform.SetCenter(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
    direction_matrix = [R[0, 0], R[1, 0], R[2, 0],
                        R[0, 1], R[1, 1], R[2, 1],
                        R[0, 2], R[1, 2], R[2, 2]]
    
    rotate_transform.SetMatrix(direction_matrix)
    centered_transform.AddTransform(rotate_transform)
    if return_direction: 
        return centered_transform, direction_matrix
    return centered_transform


def extract_connected_components(polydata):
    """ Extract connected components, return polydata with RegionId array and number of connected components"""
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(polydata)
    connect.ScalarConnectivityOn()
    connect.SetExtractionModeToAllRegions()
    connect.ColorRegionsOn()
    connect.Update()
    n_cc = connect.GetNumberOfExtractedRegions()
    return connect.GetOutput(), n_cc


def get_center_of_mass(m):
    """ Get center of mass of mesh m as numpy array"""
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(m)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = np.array(centerOfMassFilter.GetCenter())
    return center


def euclideandistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2 +
                     (point1[2] - point2[2])**2)


def numpy_to_vtk_M(nparray, name):
    vtkarray = vtk.vtkDoubleArray()
    vtkarray.SetName(name)
    vtkarray.SetNumberOfTuples(len(nparray))
    for j in range(len(nparray)):
        vtkarray.SetTuple1(j, nparray[j])
    return vtkarray


def pointthreshold(polydata, arrayname, start=0, end=1, alloff=0):
    """ Clip polydata according to given thresholds in scalar array"""
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arrayname)
    threshold.ThresholdBetween(start, end)
    if (alloff):
        threshold.AllScalarsOff()
    threshold.Update()
    surfer = vtk.vtkDataSetSurfaceFilter()
    surfer.SetInputData(threshold.GetOutput())
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

def detect_mv(lv_endo, wall, rv, max_dist_wall=5.0, factor_for_maxdist_rv=2):
    """ Detect points in the MV plane as points in LV endo far from LV wall (distance to wall > max_dist) AND far
    from the RV (to avoid getting orientations corresponding to the Aorta). Return MV polydata.
    CAREFUL: cases with holes in the LV wall segmentation (segmentation errors due to extremely thin wall,
    calcifications etc.) may wrongly detect the MV close to those holes too. Added condition related to the position of the
    connected components & added condition to keep only biggest region """

    # max_dist_wall = 5.0  # this can be left generic (independent of LV size), distance from wall
    endo_npoints = lv_endo.GetNumberOfPoints()
    mv_array = np.zeros(endo_npoints)
    np_distances_wall = np.zeros(endo_npoints)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(wall)
    locator.BuildLocator()
    for i in range(lv_endo.GetNumberOfPoints()):
        point = lv_endo.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        np_distances_wall[i] = euclideandistance(point, wall.GetPoint(closestpoint_id))
    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(np_distances_wall, 'dist_to_wall'))

    np_distances_rv = np.zeros(endo_npoints)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(rv)
    locator.BuildLocator()
    for i in range(lv_endo.GetNumberOfPoints()):
        point = lv_endo.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        np_distances_rv[i] = euclideandistance(point, rv.GetPoint(closestpoint_id))
    lv_endo.GetPointData().AddArray(
        numpy_to_vtk_M(np_distances_rv, 'dist_to_rv'))  # I will use this later for alignment wrt RV
    max_abs_dist_rv = np.max(np_distances_rv)
    max_dist_rv = np.divide(max_abs_dist_rv, factor_for_maxdist_rv)  # factor_for_maxdist_rv=2 -> more than half way from RV
    # consider a bigger region, allow to be closer to RV if there is any problem... it doesn't seem related to the size
    # but sometimes the remeshing fails and produces empty polydata
    # max_dist_rv = np.divide(max_abs_dist_rv, 2.5)

    mv_array[np.where((np_distances_wall >= max_dist_wall) & (np_distances_rv >= max_dist_rv))] = 1  # I can't do the verification of empty MV points here because I may have few points with mv_array = 1 that will create and empty surface (not connected, no cells)
    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))

    # Careful, trabeculations may be also far from LV wall and RV and will be marked as MV
    far_regions = pointthreshold(lv_endo, 'mv', 1, 1)
    if far_regions.GetNumberOfPoints() == 0:
        max_dist_rv = np.divide(max_abs_dist_rv, 3)  # Allow smaller distance. This may be needed with spherical LV where the highest distances are far from the base
        mv_array[np.where((np_distances_wall >= max_dist_wall) & (np_distances_rv >= max_dist_rv))] = 1

    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))
    far_regions = pointthreshold(lv_endo, 'mv', 1, 1)

    # get number of cc, ideally only 1 but there can be more if there are holes in wall
    far_regions_ccs, nb = extract_connected_components(far_regions)

    if nb > 1:
        # the MV is more likely to have more positive Y. Filter using y position + biggest region later
        centroids = np.zeros([nb, 3])
        for i in range(nb):
            centroids[i, :] = get_center_of_mass(pointthreshold(far_regions_ccs, 'RegionId', i, i))
        y_span = np.max(centroids[:, 1]) - np.min(centroids[:, 1])
        y_threshold = np.max(centroids[:, 1]) - np.divide(y_span, 2)

        append = vtk.vtkAppendPolyData()
        for i in range(nb):
            if centroids[i, 1] > y_threshold:
                # create new polydata only with pieces that pass the threshold
                append.AddInputData(pointthreshold(far_regions_ccs, 'RegionId', i, i))
                append.Update()

        # still get the biggest one among the ones that pass the threshold
        mv_mesh = extractlargestregion(append.GetOutput())

        # # if the previous filtering fails, check the mesh and see if directly getting the biggest cc does the work:
        # mv_mesh = extractlargestregion(far_regions)

    else:  # there is only 1, no need to extract biggest one
        mv_mesh = extractlargestregion(far_regions)

    # Update 'mv' array, keep only biggest region, I'll need it later
    mv_array = np.zeros(endo_npoints)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(lv_endo)
    locator.BuildLocator()
    for i in range(mv_mesh.GetNumberOfPoints()):
        if euclideandistance(mv_mesh.GetPoint(i), lv_endo.GetPoint(locator.FindClosestPoint(mv_mesh.GetPoint(i)))) < 0.1:
            mv_array[int(locator.FindClosestPoint(mv_mesh.GetPoint(i)))] = 1
    lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))

    return lv_endo, mv_mesh

# def detect_mv(lv_endo, wall, rv):
#     """ Detect points in the MV plane as points in LV endo far from LV wall (distance to wall > max_dist) AND far
#     from the RV (to avoid getting orientations corresponding to the Aorta). Return MV polydata. """
# 
#     max_dist_wall = 5.0  # this can be left generic (independent of LV size), distance from wall
#     endo_npoints = lv_endo.GetNumberOfPoints()
#     mv_array = np.zeros(endo_npoints)
#     np_distances_wall = np.zeros(endo_npoints)
#     locator = vtk.vtkPointLocator()
#     locator.SetDataSet(wall)
#     locator.BuildLocator()
#     for i in range(lv_endo.GetNumberOfPoints()):
#         point = lv_endo.GetPoint(i)
#         closestpoint_id = locator.FindClosestPoint(point)
#         np_distances_wall[i] = euclideandistance(point, wall.GetPoint(closestpoint_id))
#     lv_endo.GetPointData().AddArray(numpy_to_vtk_M(np_distances_wall, 'dist_to_wall'))
# 
#     np_distances_rv = np.zeros(endo_npoints)
#     locator = vtk.vtkPointLocator()
#     locator.SetDataSet(rv)
#     locator.BuildLocator()
#     for i in range(lv_endo.GetNumberOfPoints()):
#         point = lv_endo.GetPoint(i)
#         closestpoint_id = locator.FindClosestPoint(point)
#         np_distances_rv[i] = euclideandistance(point, rv.GetPoint(closestpoint_id))
#     lv_endo.GetPointData().AddArray(numpy_to_vtk_M(np_distances_rv, 'dist_to_rv'))  # I will use this later for alignment wrt RV
#     max_abs_dist_rv = np.max(np_distances_rv)
#     max_dist_rv = np.divide(max_abs_dist_rv, 2)  # more than half way from RV
# 
#     mv_array[np.where((np_distances_wall >= max_dist_wall) & (np_distances_rv >= max_dist_rv))] = 1   # I can't do the verification of empty MV points here because I may have few points with mv_array = 1 that will create and empty surface (not connected, no cells)
#     lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))
# 
#     # Careful, trabeculations may be also far from LV wall and RV and will be marked as MV
#     far_regions = pointthreshold(lv_endo, 'mv', 1, 1)
#     if far_regions.GetNumberOfPoints() == 0:
#         max_dist_rv = np.divide(max_abs_dist_rv, 3)  # Allow smaller distance. This may be needed with spherical LV where the highest distances are far from the base
#         mv_array[np.where((np_distances_wall >= max_dist_wall) & (np_distances_rv >= max_dist_rv))] = 1
# 
#     lv_endo.GetPointData().AddArray(numpy_to_vtk_M(mv_array, 'mv'))
#     far_regions = pointthreshold(lv_endo, 'mv', 1, 1)
#     mv_mesh = extractlargestregion(far_regions)
#     return lv_endo, mv_mesh


def cellnormals(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.Update()
    return normals.GetOutput()


def get_vertices(polydata):
    npoints = polydata.GetNumberOfPoints()
    v = np.zeros([npoints, 3])
    for i in range(npoints):
        v[i, :] = np.array(polydata.GetPoint(i))
    return v


def set_vertices(polydata, v):
    points = vtk.vtkPoints()
    npoints = v.shape[0]
    points.SetNumberOfPoints(npoints)
    for i in range(npoints):
        points.SetPoint(i, v[i, :])
    polydata.SetPoints(points)
    return polydata


## Main function ####################################################
def orientation(input_paths=None,
    input_imgs=None,
    n_clusters=2000,
    ref_size=512,
    threshold=None,
    write_suffix='SAX',
    ):
    """
    :param input_imgs: {"intensity", "lvendo", "lvwall", "rvepi"}
    :type input_imgs: dict
    """
    # load input images
    if input_imgs is None: 
        input_imgs = {key: read_sitk(_path) for key, _path in input_paths.items()}

    if threshold is not None: 
        for key in input_imgs.keys():
            if key == 'intensity': 
                continue

            if isinstance(threshold, dict): 
                val_min = threshold['val_min']
                val_max = threshold['val_max']

            elif isinstance(threshold, (list, tuple)):
                val_min, val_max = threshold

            else: 
                val_min = threshold
                val_max = 1
                
            input_imgs[key] = sitk.BinaryThreshold(input_imgs[key], 
                    lowerThreshold=val_min, 
                    upperThreshold=val_max,
                    insideValue=1,
                    outsideValue=0
                    )
    lv_endo_img = input_imgs['lvendo']
    rv_epi_img = input_imgs['rvepi']

    if not input_imgs.get('lvwall'): 
        lv_epi_img = input_imgs['lvepi']
        lv_wall_img = lv_epi_img - lv_endo_img
        input_imgs['lvwall'] = lv_wall_img
    lv_wall_img = input_imgs['lvwall']
    
    if input_imgs.get('intensity'):
        im = input_imgs['intensity']
        min_intensity_value = np.min(sitk.GetArrayViewFromImage(im))
    else: 
        im = lv_endo_img
    
    dimension = im.GetDimension()
    reference_direction = np.identity(dimension).flatten()
    
    reference_origin = np.zeros(dimension)  # (0,0,0) Origin
    reference_size = [ref_size] * dimension
    reference_physical_size = np.zeros(dimension)
    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in zip(im.GetSize(), im.GetSpacing(), reference_physical_size)]
    reference_spacing = [0.5, 0.5, 0.5] # set spacing to isotropically 0.5mm
    
    reference_image = sitk.Image(reference_size, im.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))  # geometrical center (coordinates)
    
    # get surface mesh
    lv_endo_m, lv_wall_m, rv_epi_m = [
            utils.marching_cubes(_img, closing_radius=1, n_clusters=n_clusters)
        for _img in [lv_endo_img, lv_wall_img, rv_epi_img]]
    ##########      1. Align MV to theoretical MV plane      ##########
    # Detect MV and MV centroid. Find points in the endo that are far from points in the LV wall.
    lv_endo_m, mv_m = detect_mv(lv_endo_m, lv_wall_m, rv_epi_m, max_dist_wall=5.0, factor_for_maxdist_rv=2.0)  # Only MV (aprox)
    mv_normals_m = cellnormals(mv_m)
    mv_normals = vtk_to_numpy(mv_normals_m.GetCellData().GetArray('Normals'))
    mv_normal = np.mean(mv_normals, axis=0)
    mv_centroid = get_center_of_mass(mv_m)
    
    # Find apex id as the furthest point to mv_centroid (I will use it in the latest alignment)
    np_distances_mv = np.zeros(lv_endo_m.GetNumberOfPoints())
    for i in range(lv_endo_m.GetNumberOfPoints()):
        np_distances_mv[i] = euclideandistance(mv_centroid, lv_endo_m.GetPoint(i))
    lv_endo_m.GetPointData().AddArray(numpy_to_vtk_M(np_distances_mv, 'dist_to_MV_centroid'))
    apex_id = np.argmax(np_distances_mv)
    # print('Apex id according to distances ', apex_id)
    # # Get initial LV long axis (just to draw the plane in the paper)
    # p0_aux = mv_centroid
    # p1_aux = np.array(lv_endo_m.GetPoint(apex_id))
    # v_lax = np.divide(p1_aux - p0_aux, np.linalg.norm(p1_aux - p0_aux))  # get unit vector, normalize
    
    # Find rotation matrix that will align the MV plane to theoretical MV plane
    #print('Aligning MV. Computing rotation matrix...')
    v1 = - np.divide(mv_normal, np.linalg.norm(mv_normal))  # Get unit vector, normalize. Use opposite of the normal
    # (normals point towards the outside) and positive normal of the XY plane.
    v2 = np.array([0, 0, 1])     # Theoretical MV normal, short axis plane, XY plane
    
    # Given v1 and v2 vectors, find rotation matrix that aligns v1 to v2.
    # Adapted from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = np.cross(v1, v2)
    # s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R1 = np.eye(3) + Vx + np.dot(Vx, Vx) * np.divide(1, 1 + c)
    
    # Apply transformation to LV endo mesh
    lv_vertices_ori = get_vertices(lv_endo_m).T
    lv_vertices_rotated = np.dot(R1, lv_vertices_ori)
    m_lv_vertices_rotated = set_vertices(lv_endo_m, lv_vertices_rotated.T)
    
    # Rotate also RV. I will need it later for subsequent alignment
    rv_vertices_ori = get_vertices(rv_epi_m).T
    rv_vertices_rotated = np.dot(R1, rv_vertices_ori)
    m_rv_vertices_rotated = set_vertices(rv_epi_m, rv_vertices_rotated.T)
    
    ##########      Align LV septum. Find vector within the MV surface that points from LV to RV      ##############
    mv_rotated_m = pointthreshold(m_lv_vertices_rotated, 'mv', 1, 1)
    # remesh to increase spatial resolution
    m_mv_rotated_remeshed = utils.uniform_meshing(mv_rotated_m, n_clusters=2000, subdivide=True) # need subdivide in this case
    
    p_center_mv = get_center_of_mass(m_mv_rotated_remeshed)
    # Compute distances to center of mass to keep only closest ones
    np_distances_center = np.zeros(m_mv_rotated_remeshed.GetNumberOfPoints())
    for i in range(m_mv_rotated_remeshed.GetNumberOfPoints()):
        np_distances_center[i] = euclideandistance(m_mv_rotated_remeshed.GetPoint(i), p_center_mv)
    
    m_mv_rotated_remeshed.GetPointData().AddArray(numpy_to_vtk_M(np_distances_center, 'dist_to_center'))
    _max_distance = 5
    center_poly = pointthreshold(m_mv_rotated_remeshed, 'dist_to_center', 0, _max_distance)

    while center_poly.GetNumberOfPoints() == 0:
        # relax threshold distance
        _max_distance+=1
        center_poly = pointthreshold(m_mv_rotated_remeshed, 'dist_to_center', 0, _max_distance)

    # Re-compute distances to RV, now only for the center_poly
    np_distances_rv = np.zeros(center_poly.GetNumberOfPoints())
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(m_rv_vertices_rotated)
    locator.BuildLocator()
    for i in range(center_poly.GetNumberOfPoints()):
        point = center_poly.GetPoint(i)
        closestpoint_id = locator.FindClosestPoint(point)
        np_distances_rv[i] = euclideandistance(point, m_rv_vertices_rotated.GetPoint(closestpoint_id))
    
    center_poly.GetPointData().AddArray(numpy_to_vtk_M(np_distances_rv, 'dist_to_rv'))
    p0_id = np.argmin(np_distances_rv)
    p1_id = np.argmax(np_distances_rv)
    p0 = np.array(center_poly.GetPoint(p0_id))
    p1 = np.array(center_poly.GetPoint(p1_id))
    
    v11 = np.divide(p1 - p0, np.linalg.norm(p1 - p0))  # get unit vector, normalize
    v21 = np.array([1, 0, 0])  # unit vector within MV plane and pointing from RV to LV
    
    v = np.cross(v11, v21)
    # s = np.linalg.norm(v)
    c = np.dot(v11, v21)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R2 = np.eye(3) + Vx + np.dot(Vx, Vx) * np.divide(1, 1 + c)
    
    # Apply the 2 rotations at the time -> R2 * (R * Vertices)
    #print('Aligning LV with regard to RV...')
    R2_R1 = np.dot(R2, R1)
    vertices_lv_rotated_twice = np.dot(R2_R1, lv_vertices_ori)
    m_vertices_lv_rotated_twice = set_vertices(lv_endo_m, vertices_lv_rotated_twice.T)
    
    
    ##########      Finally, align LV long axis (i.e. line from center of MV to LV apex)      ##########
    mv_centroid = get_center_of_mass(pointthreshold(m_vertices_lv_rotated_twice, 'mv', 1, 1))
    #print('Aligning LV long axis...')
    p0 = mv_centroid
    p1 = np.array(m_vertices_lv_rotated_twice.GetPoint(apex_id))
    
    v1 = np.divide(p1 - p0, np.linalg.norm(p1 - p0))  # get unit vector, normalize
    v2 = np.array([0, 0, 1])  # Theoretical LV long axis
    v = np.cross(v1, v2)
    # s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R3 = np.eye(3) + Vx + np.dot(Vx, Vx) * np.divide(1, 1 + c)
    
    # Now, the 3 rotations should be applied as: R3 * (R2 * (R * vertices) )
    R_aux = np.dot(R2, R1)
    R_final = np.dot(R3, R_aux)
    
    # Resample CT image and initial masks using the transformation given by appropriate translation + R_final
    centered_transform = get_rotation_transform(R=R_final,
            img=lv_endo_img,
            reference_image=reference_image,
            reference_origin=reference_origin,
            reference_center=reference_center,
            default_pixel_value=0,
            )

    ## 4th Rotation 
    lvendo_rot = sitk.Resample(lv_endo_img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    # rv_im = sitk.ReadImage(rv_epi_filename)
    rvepi_rot = sitk.Resample(rv_epi_img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    r_lvendo = sitk.GetArrayFromImage(lvendo_rot)
    r_rvepi = sitk.GetArrayFromImage(rvepi_rot)

    start = None
    end = None
    for i, mask2d in enumerate(r_lvendo):
        n = mask2d.sum()
        if start is None and n != 0:
            start = i
        if start is not None and n == 0:
            end = i
            break

    if end == None:
        end = r_lvendo.shape[0]    # z in ITK
    index = start + (end - start) // 2  # midway along the long axis

    lvcenter = np.array(np.nonzero(r_lvendo[index])).mean(axis=1)
    rvcenter = np.array(np.nonzero(r_rvepi[index])).mean(axis=1)

    direction = rvcenter - lvcenter
    angle = np.arctan2(direction[1], direction[0]) + np.pi / 2

    last_rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                              [np.sin(angle), np.cos(angle), 0],
                              [0, 0, 1],
                              ])
    R_final2 = np.dot(last_rotation, R_final)               # add the 4th rotation to previous rotation matrix

    # get new transform funciton with R_final2
    centered_transform, direction_matrix = get_rotation_transform(R=R_final2,
            img=lv_endo_img,
            reference_image=reference_image,
            reference_origin=reference_origin,
            reference_center=reference_center,
            default_pixel_value=0,
            return_direction=True,
            )
    
    # Resample sitk image
    return_imgs = {}
    for key, _im in input_imgs.items():
        if key == "intensity":
            sampled_img = sitk.Resample(_im, reference_image, centered_transform, sitk.sitkLinear, int(min_intensity_value))
            sampled_img = sitk.Cast(sampled_img, sitk.sitkInt16)

        else: 
            # sampled_img = sitk.Resample(_im, reference_image, centered_transform, sitk.sitkLinear, 0.0)
            sampled_img = sitk.Resample(_im, reference_image, centered_transform, sitk.sitkNearestNeighbor, 0.0)
            sampled_img = sitk.Cast(sampled_img, sitk.sitkUInt8)
        sampled_img.SetDirection(np.array(direction_matrix).flatten())
        # copy metadata
        for meta_key in im.GetMetaDataKeys():
            sampled_img.SetMetaData(meta_key, im.GetMetaData(meta_key))
        return_imgs[key] = sampled_img

    if _max_distance != 5: 
        print(f"Relax mv thresholding to {_max_distance}")
    return return_imgs
