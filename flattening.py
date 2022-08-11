import os, math

import pyvista as pv
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as linalg_sp
from scipy.sparse import vstack, hstack, coo_matrix, csc_matrix
import SimpleITK as sitk
import skimage.draw as skdraw
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import utils

def writevtk(surface, filename, type='ascii'):
    """Write binary or ascii VTK file"""
    writer = vtk.vtkPolyDataWriter()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        writer.SetInputData(surface)
    else:
        writer.SetInput(surface)
    writer.SetFileName(filename)
    if type == 'ascii':
        writer.SetFileTypeToASCII()
    elif type == 'binary':
        writer.SetFileTypeToBinary()
    writer.Write()

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


def get_center_of_mass(m):
    """ Get center of mass of mesh m as numpy array"""
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(m)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = np.array(centerOfMassFilter.GetCenter())
    return center


def extractboundaryedge(polydata):
    edge = vtk.vtkFeatureEdges()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        edge.SetInputData(polydata)
    else:
        edge.SetInput(polydata)
    edge.FeatureEdgesOff()
    edge.NonManifoldEdgesOff()
    edge.Update()
    return edge.GetOutput()


def get_ordered_cont_ids_based_on_distance(mesh):
    """ Given a contour, get the ordered list of Ids (not ordered by default).
    Open the mesh duplicating the point with id = 0. Compute distance transform of point 0
    and get a ordered list of points (starting in 0) """
    m = vtk.vtkMath()
    m.RandomSeed(0)
    # copy the original mesh point by point
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    cover = vtk.vtkPolyData()
    nver = mesh.GetNumberOfPoints()
    points.SetNumberOfPoints(nver+1)

    new_pid = nver  # id of the duplicated point
    added = False

    for j in range(mesh.GetNumberOfCells()):
        # get the 2 point ids
        ptids = mesh.GetCell(j).GetPointIds()
        cell = mesh.GetCell(j)
        if (ptids.GetNumberOfIds() != 2):
            # print "Non contour mesh (lines)"
            break

        # read the 2 involved points
        pid0 = ptids.GetId(0)
        pid1 = ptids.GetId(1)
        p0 = mesh.GetPoint(ptids.GetId(0))   # returns coordinates
        p1 = mesh.GetPoint(ptids.GetId(1))

        if pid0 == 0:
            if added == False:
                # Duplicate point 0. Add gaussian noise to the original point
                new_p = [p0[0] + m.Gaussian(0.0, 0.0005), p0[1] + m.Gaussian(0.0, 0.0005), p0[2] + m.Gaussian(0.0, 0.0005)]
                points.SetPoint(new_pid, new_p)
                points.SetPoint(pid1, p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(pid1)
                polys.InsertCellPoint(new_pid)
                added = True
            else:  # act normal
                points.SetPoint(ptids.GetId(0), p0)
                points.SetPoint(ptids.GetId(1), p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))
        elif pid1 == 0:
            if added == False:
                new_p = [p1[0] + m.Gaussian(0.0, 0.0005), p1[1] + m.Gaussian(0.0, 0.0005), p1[2] + m.Gaussian(0.0, 0.0005)]
                points.SetPoint(new_pid, new_p)
                points.SetPoint(pid0, p0)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(pid0)
                polys.InsertCellPoint(new_pid)
                added = True
            else:  # act normal
                points.SetPoint(ptids.GetId(0), p0)
                points.SetPoint(ptids.GetId(1), p1)
                polys.InsertNextCell(2)
                polys.InsertCellPoint(cell.GetPointId(0))
                polys.InsertCellPoint(cell.GetPointId(1))

        else:
            points.SetPoint(ptids.GetId(0), p0)
            points.SetPoint(ptids.GetId(1), p1)
            polys.InsertNextCell(2)
            polys.InsertCellPoint(cell.GetPointId(0))
            polys.InsertCellPoint(cell.GetPointId(1))

    if added == False:
        print('Warning: I have not added any point, list of indexes may not be correct.')
    cover.SetPoints(points)
    cover.SetPolys(polys)
    if not vtk.vtkVersion.GetVTKMajorVersion() > 5:
        cover.Update()

    # workaround triangluate
    cover = utils.path_triagulate(cover)
    
    # compute distance from point with id 0 to all the rest
    npoints = cover.GetNumberOfPoints()
    dists = np.zeros(npoints)
    for i in range(npoints):
        [dists[i], poly] = compute_geodesic_distance(cover, int(0), i)
    list_ = np.argsort(dists).astype(int)
    return list_[0:len(list_)-1]    # skip last one, duplicated


def ExtractVTKPoints(mesh):
    """Extract points from vtk structures. Return the Nx3 numpy.array of the vertices."""
    n = mesh.GetNumberOfPoints()
    vertex = np.zeros((n, 3))
    for i in range(n):
        mesh.GetPoint(i, vertex[i, :])
    return vertex


def ExtractVTKTriFaces(mesh):
    """Extract triangular faces from vtkPolyData. Return the Nx3 numpy.array of the faces (make sure there are only triangles)."""
    m = mesh.GetNumberOfCells()
    faces = np.zeros((m, 3), dtype=int)
    for i in range(m):
        ptIDs = vtk.vtkIdList()
        mesh.GetCellPoints(i, ptIDs)
        if ptIDs.GetNumberOfIds() != 3:
            raise Exception("Nontriangular cell!")
        faces[i, 0] = ptIDs.GetId(0)
        faces[i, 1] = ptIDs.GetId(1)
        faces[i, 2] = ptIDs.GetId(2)
    return faces


def ComputeLaplacian(vertex, faces):
    """Calculates the laplacian of a mesh
    vertex 3xN numpy.array: vertices
    faces 3xM numpy.array: faces"""
    n = vertex.shape[1]
    m = faces.shape[1]

    # compute mesh weight matrix
    W = sparse.coo_matrix((n, n))
    for i in np.arange(1, 4, 1):
        i1 = np.mod(i - 1, 3)
        i2 = np.mod(i, 3)
        i3 = np.mod(i + 1, 3)
        pp = vertex[:, faces[i2, :]] - vertex[:, faces[i1, :]]
        qq = vertex[:, faces[i3, :]] - vertex[:, faces[i1, :]]
        # normalize the vectors
        pp = pp / np.sqrt(np.sum(pp ** 2, axis=0))
        qq = qq / np.sqrt(np.sum(qq ** 2, axis=0))

        # compute angles
        ang = np.arccos(np.sum(pp * qq, axis=0))
        W = W + sparse.coo_matrix((1 / np.tan(ang), (faces[i2, :], faces[i3, :])), shape=(n, n))
        W = W + sparse.coo_matrix((1 / np.tan(ang), (faces[i3, :], faces[i2, :])), shape=(n, n))

    # compute laplacian
    d = W.sum(axis=0)
    D = sparse.dia_matrix((d, 0), shape=(n, n))
    L = D - W
    return L


def flat_w_constraints(m, boundary_ids, constraints_ids, x0_b, y0_b, x0_c, y0_c):
    """ Conformal flattening fitting boundary points to (x0_b,y0_b) coordinate positions
    and additional contraint points to (x0_c,y0_c).
    Solve minimization problem using quadratic programming: https://en.wikipedia.org/wiki/Quadratic_programming"""

    penalization = 1000
    vertex = ExtractVTKPoints(m).T    # 3 x n_vertices
    faces = ExtractVTKTriFaces(m).T
    n = vertex.shape[1]
    L = ComputeLaplacian(vertex, faces)
    L = L.tolil()
    L[boundary_ids, :] = 0.0     # Not conformal there
    for i in range(boundary_ids.shape[0]):
         L[boundary_ids[i], boundary_ids[i]] = 1

    L = L*penalization

    Rx = np.zeros(n)
    Ry = np.zeros(n)
    Rx[boundary_ids] = x0_b * penalization
    Ry[boundary_ids] = y0_b * penalization

    L = L.tocsr()
    # result = np.zeros((Rx.size, 2))

    nconstraints = constraints_ids.shape[0]
    M = np.zeros([nconstraints, n])   # M, zero rows except 1 in constraint point
    for i in range(nconstraints):
        M[i, constraints_ids[i]] = 1
    dx = x0_c
    dy = y0_c

    block1 = hstack([L.T.dot(L), M.T])

    zeros_m = coo_matrix(np.zeros([len(dx),len(dx)]))
    block2 = hstack([M, zeros_m])

    C = vstack([block1, block2])

    prodx = coo_matrix([L.T.dot(Rx)])
    dxx = coo_matrix([dx])
    cx = hstack([prodx, dxx])

    prody = coo_matrix([L.T.dot(Ry)])
    dyy = coo_matrix([dy])
    cy = hstack([prody, dyy])

    solx = linalg_sp.spsolve(C, cx.T)
    soly = linalg_sp.spsolve(C, cy.T)

    # print('There are: ', len(np.argwhere(np.isnan(solx))), ' nans')
    # print('There are: ', len(np.argwhere(np.isnan(soly))), ' nans')
    if len(np.argwhere(np.isnan(solx))) > 0:
        print('WARNING!!! matrix is singular. It is probably due to the convergence of 2 different division lines in the same point.')
        print('Trying to assign different 2D possition to same 3D point. Try to create new division lines or increase resolution of mesh.')

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()

    pts.SetNumberOfPoints(n)
    for i in range(n):
        pts.SetPoint(i, solx[i], soly[i], 0)

    pd.SetPoints(pts)
    pd.SetPolys(m.GetPolys())
    pd.Modified()
    return pd


def compute_geodesic_distance(mesh, id_p1, id_p2):
    """Compute geodesic distance from point id_p1 to id_p2 on surface 'mesh'
    It first computes the path across the edges and then the corresponding distance adding up point to point distances)"""
    path = find_create_path(mesh, id_p1, id_p2)
    total_dist = 0
    n = path.GetNumberOfPoints()
    for i in range(n-1):   # Ids are ordered in the new polydata, from 0 to npoints_in_path
        p0 = path.GetPoint(i)
        p1 = path.GetPoint(i+1)
        dist = math.sqrt(math.pow(p0[0]-p1[0], 2) + math.pow(p0[1]-p1[1], 2) + math.pow(p0[2]-p1[2], 2) )
        total_dist = total_dist + dist
    return total_dist, path


def find_create_path(mesh, p1, p2):
    """Get shortest path (using Dijkstra algorithm) between p1 and p2 on the mesh. Returns a polydata"""
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    if vtk.vtkVersion().GetVTKMajorVersion() > 5:
        dijkstra.SetInputData(mesh)
    else:
        dijkstra.SetInput(mesh)
    dijkstra.SetStartVertex(p1)
    dijkstra.SetEndVertex(p2)
    dijkstra.Update()
    return dijkstra.GetOutput()


def euclideandistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 +
             (point1[1] - point2[1])**2 +
             (point1[2] - point2[2])**2)


def transfer_all_scalar_arrays_by_point_id(m1, m2):
    """ Transfer all scalar arrays from m1 to m2 by point id"""
    for i in range(m1.GetPointData().GetNumberOfArrays()):
        print('Transferring scalar array: {}'.format(m1.GetPointData().GetArray(i).GetName()))
        m2.GetPointData().AddArray(m1.GetPointData().GetArray(i))
        
        
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


def flip_disk(mesh):
    """Flip the disk along the y axis; difference between seeing the LV from the inside or from the outside"""
    m_vertices_ori = get_vertices(mesh)

    m_vertices_flip = np.array([m_vertices_ori[:, 0], -m_vertices_ori[:, 1], m_vertices_ori[:, 2]])
    return set_vertices(mesh, m_vertices_flip.T)


def split_base(mesh):
    """
    """
    m = mesh
    # np_aha = vtk_to_numpy(m.GetPointData().GetArray('aha'))
    np_thickness = vtk_to_numpy(m.GetPointData().GetArray('thickness'))

    np_pot_base = np.zeros(np_thickness.shape)
    np_pot_base[np_thickness == 0] = 1
    pot_base_array = numpy_to_vtk(np_pot_base)
    pot_base_array.SetName('potential_base')
    m.GetPointData().AddArray(pot_base_array)

    potential_bases, nb = extract_connected_components(pointthreshold(m, 'potential_base', 1, 1))
    # potential_bases, nb = extract_connected_components(pointthreshold(m, 'potential_base', 1, 1))

    if nb > 1:
        # the MV is more likely to have smaller z in SAX. Filter using y position + biggest region later
        centroids = np.zeros([nb, 3])
        for i in range(nb):
            _p_base = pointthreshold(potential_bases, 'RegionId', i, i)
            if _p_base.GetNumberOfPoints() > 0:
                centroids[i, :] = get_center_of_mass(_p_base)
            else:
                centroids[i, :] = np.nan

        _centroids = centroids[~np.isnan(centroids[:, 0])]
        z_span = np.max(_centroids[:, 2]) - np.min(_centroids[:, 2])
        z_threshold = np.max(_centroids[:, 2]) - np.divide(z_span, 2)

        append = vtk.vtkAppendPolyData()
        for i in range(nb):
            if (centroids[i, 2] <= z_threshold) and (centroids[i, 2] != np.nan):
                # create new polydata only with pieces that pass the threshold
                append.AddInputData(pointthreshold(potential_bases, 'RegionId', i, i))
                append.Update()

        # still get the biggest one among the ones that pass the threshold
        base = extractlargestregion(append.GetOutput())

    else:  # there is only 1, no need to extract biggest one
        base = extractlargestregion(pointthreshold(m, 'potential_base', 1, 1)) 

    np_base = np.zeros(np_thickness.shape)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(m)
    locator.BuildLocator()
    for i in range(base.GetNumberOfPoints()):
        p = base.GetPoint(i)
        closest_p = locator.FindClosestPoint(p)
        if euclideandistance(p, m.GetPoint(closest_p)) < 0.01:
            np_base[closest_p] = 1

    base_array = numpy_to_vtk(np_base)
    base_array.SetName('base')
    m.GetPointData().AddArray(base_array)

    # writevtk(m, m_aux_filename)

    m_no_base = extractlargestregion(pointthreshold(m, 'base', 0, 0))
    m_base = pointthreshold(m, 'base', 1, 1)

    m_no_base.GetPointData().RemoveArray('base')
    m_no_base.GetPointData().RemoveArray('potential_base')
    m_no_base.GetPointData().RemoveArray('hectic_base')
    return pv.wrap(m_no_base), pv.wrap(m_base)


def get_bullseye_mesh(
        midwall_mesh,

        rdisk = 1.0,
        ap_x0 = np.array([0.0]),
        ap_y0 = np.array([0.0]),

        visualize_from_outside = True,  

        n = 1/3.0,
        ):
    """
    :param visualize_from_outside: visualize LV from the inside (AHA) or from the outside
    :type visualize_from_outside:

    :param n: radial displacement exponent value, (control the distribution of points on the flatten mesh).
    :type n: float
    """
    # detect the base as the BIGGEST CONNECTED COMPONENT (make only 2 hole) with thickness = 0, aha = 0, AND lowest z centroid
    
    # m_no_base, m_base = split_base(mesh=midwall_mesh)

    # Find seeds needed for the flattening
    locator_no_base = vtk.vtkPointLocator()
    locator_no_base.SetDataSet(midwall_mesh)
    locator_no_base.BuildLocator()

    # id apex
    apex = pointthreshold(midwall_mesh, 'regions', 17, 17)   # aha probed contains errors in the region borders, use 'regions'
    center = get_center_of_mass(apex)
    id_ap = int(locator_no_base.FindClosestPoint(center))
    # print(id_ap)

    # Detect edges -> base contour
    cont = extractboundaryedge(midwall_mesh)
    edge_cont_ids = get_ordered_cont_ids_based_on_distance(cont).astype(int)     # warnings here

    # find corresponding ordered points in the COMPLETE mesh. Use same locator as before
    cont_base_ids = np.zeros(edge_cont_ids.shape[0]).astype(int) - 1
    for i in range(cont_base_ids.shape[0]):
        p = cont.GetPoint(edge_cont_ids[i])
        cont_base_ids[i] = locator_no_base.FindClosestPoint(p)
    # print(cont_base_ids)

    # Find point at the base and at the interface between region 2 and 3.
    # Get largest region and not only the contour to avoid puntual errors...
    piece2 = extractlargestregion(pointthreshold(midwall_mesh, 'regions', 2, 2))
    piece3 = extractlargestregion(pointthreshold(midwall_mesh, 'regions', 3, 3))
    piece4 = extractlargestregion(pointthreshold(midwall_mesh, 'regions', 4, 4))

    locator_piece2 = vtk.vtkPointLocator()
    locator_piece2.SetDataSet(piece2)
    locator_piece2.BuildLocator()
    locator_piece4 = vtk.vtkPointLocator()
    locator_piece4.SetDataSet(piece4)
    locator_piece4.BuildLocator()
    dist_array1 = np.zeros(piece3.GetNumberOfPoints())
    dist_array2 = dist_array1.copy()
    for i in range(piece3.GetNumberOfPoints()):
        p = piece3.GetPoint(i)
        closest_p_inpiece2 = locator_piece2.FindClosestPoint(p)
        closest_p_inpiece4 = locator_piece4.FindClosestPoint(p)
        dist_array1[i] = euclideandistance(p, piece2.GetPoint(closest_p_inpiece2))
        dist_array2[i] = euclideandistance(p, piece4.GetPoint(closest_p_inpiece4))

    array1 = numpy_to_vtk(dist_array1)
    array1.SetName('dist_to_piece2')
    array2 = numpy_to_vtk(dist_array2)
    array2.SetName('dist_to_piece4')
    piece3.GetPointData().AddArray(array1)
    piece3.GetPointData().AddArray(array2)

    piece3_contour = extractboundaryedge(piece3)
    # detect base border/contour in this piece border
    np_cont_array = np.zeros(piece3_contour.GetNumberOfPoints())
    locator_cont = vtk.vtkPointLocator()
    locator_cont.SetDataSet(cont)
    locator_cont.BuildLocator()
    for i in range(piece3_contour.GetNumberOfPoints()):
        p = piece3_contour.GetPoint(i)
        if euclideandistance(p, cont.GetPoint(locator_cont.FindClosestPoint(p))) < 0.01:
            np_cont_array[i] = 1
    cont_array = numpy_to_vtk(np_cont_array)
    cont_array.SetName('base_cont')
    piece3_contour.GetPointData().AddArray(cont_array)
    #writevtk(piece3_contour, path + pat_id + '/' + name + '-piece3-cont.vtk')
    border = pointthreshold(piece3_contour, 'base_cont', 1, 1)   # keep only the region in the base

    dist_border3_to_piece2 = border.GetPointData().GetArray('dist_to_piece2')
    dist_border3_to_piece4 = border.GetPointData().GetArray('dist_to_piece4')
    p_0 = border.GetPoint(int(np.argmin(dist_border3_to_piece2)))
    p_aux = border.GetPoint(int(np.argmin(dist_border3_to_piece4)))

    # in surface
    id_base0 = locator_no_base.FindClosestPoint(p_0)
    id_base_aux = locator_no_base.FindClosestPoint(p_aux)
    # print(id_base0)
    # print(id_base_aux)

    # base -> external disk
    # order cont_base_ids to start in id_base0
    reordered_base_cont = np.append(cont_base_ids[int(np.where(cont_base_ids == id_base0)[0]): cont_base_ids.shape[0]],
                                  cont_base_ids[0: int(np.where(cont_base_ids == id_base0)[0])]).astype(int)

    # print('Reordered base', reordered_base_cont)
    segment_length_aprox = len(reordered_base_cont)/6

    # check if the list of ordered points corresponding to the base contours has to be flipped
    pos_auxpoint = int(np.where(reordered_base_cont == id_base_aux)[0])
    if pos_auxpoint > 2*segment_length_aprox:     # use the length of 2 segments to allow some error
        # Flip
        print('I ll flip the base ids')
        aux = np.zeros(reordered_base_cont.size)
        for i in range(reordered_base_cont.size):
            aux[reordered_base_cont.size - 1 - i] = reordered_base_cont[i]
        reordered_base_cont = np.append(aux[aux.size - 1], aux[0:aux.size - 1]).astype(int)

    complete_circumf_t = np.linspace(np.pi, np.pi + 2*np.pi, len(reordered_base_cont), endpoint=False)  # starting in pi
    x0_ext = np.cos(complete_circumf_t) * rdisk
    y0_ext = np.sin(complete_circumf_t) * rdisk

    m_disk = flat_w_constraints(midwall_mesh, reordered_base_cont, np.array([id_ap]), x0_ext, y0_ext, ap_x0.astype(float),
                                ap_y0.astype(float))

    transfer_all_scalar_arrays_by_point_id(midwall_mesh, m_disk)
    # writevtk(m_disk, mesh_ofilename2)


    # Apply Bruno's radial displacement to enlarge central part and get more uniform mesh
    # Paun, Bruno, et al. "Patient independent representation of the detailed cardiac ventricular anatomy."
    # Medical image analysis (2017)
    npoints = m_disk.GetNumberOfPoints()
    m_out = vtk.vtkPolyData()
    points = vtk.vtkPoints()

    points.SetNumberOfPoints(npoints)
    for i in range(npoints):
        q = np.array(m_disk.GetPoint(i))
        p = np.copy(q)
        x = math.pow(math.pow(p[0], 2) + math.pow(p[1], 2), np.divide(n, 2.0))*math.cos(math.atan2(p[1], p[0]))
        y = math.pow(math.pow(p[0], 2) + math.pow(p[1], 2), np.divide(n, 2.0))*math.sin(math.atan2(p[1], p[0]))
        points.SetPoint(i, x, y, 0)
    m_out.SetPoints(points)
    m_out.SetPolys(m_disk.GetPolys())

    transfer_all_scalar_arrays_by_point_id(m_disk, m_out)
    if visualize_from_outside:
        m_out = flip_disk(m_out)

    return m_out


def flat_mesh_to_np(mesh, 
        output_shape,
        pad=1, 
        scalar_name="thickness", 
        return_mask=True,
        ):
    """ Convert flatten mesh to numpy array.
    """
    # get circle mask
    center = output_shape//2
    radius = center - pad
    shape = (output_shape, output_shape)
    rr, cc = skdraw.disk((center, center), 
            radius=radius, 
            shape=shape)
    blank = np.zeros(shape)
    rr_norm = (rr - center)/radius
    cc_norm = (cc - center)/radius

    # set up probe filter
    probe_points = vtk.vtkPoints()
    for r, c in zip(rr_norm, cc_norm):
        probe_points.InsertNextPoint((r, c, 0))
    probe_polydata = vtk.vtkPolyData()
    probe_polydata.SetPoints(probe_points)
    probe_polydata.Modified()

    probe_filt = vtk.vtkProbeFilter()
    probe_filt.SetInputData(probe_polydata)
    probe_filt.SetSourceData(mesh)
    probe_filt.Update()
    output = probe_filt.GetOutput()
    val_array = pv.wrap(output)[scalar_name]
    for x, y, val in zip(rr, cc, val_array):
        blank[x, y] = val
    
    if return_mask:
        mask = np.zeros_like(blank)
        for x, y in zip(rr, cc):
            mask[x, y] = 1
        return blank, mask
    return blank


def np_to_mesh(
        array,
        disk_mesh, 
        surface_mesh, 
        array_name='remap',
        array_pad=1,
        ):
    """ Remap numpy array to disk_mesh.
    """
    shape = array.shape[0]
    center = shape//2
    radius = center - array_pad

    n_points = disk_mesh.GetNumberOfPoints()
    remap_array = []
    for ind in range(n_points): 
        # get pos 
        x_pos, y_pos, _ = disk_mesh.GetPoint(ind)
        
        # get x, y in index
        x = int((x_pos+1)*radius)
        y = int((y_pos+1)*radius)
        remap_array.append(array[x, y])
    disk_mesh = pv.wrap(disk_mesh)  
    surface_mesh = pv.wrap(surface_mesh)

    disk_mesh[array_name] = remap_array
    surface_mesh[array_name] = remap_array
    return disk_mesh, surface_mesh
