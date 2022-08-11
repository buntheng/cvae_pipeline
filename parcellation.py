
import numpy as np 
import SimpleITK as sitk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import utils

def lvwall_aha_parcellation(
        lvwall_thickness, 
        lvendo_mask,
        lvepi_mask,
        dilate_wall=True,
        dilate_mm=1.0,
        ):
    """
    """
    np_lvwall_thickness = sitk.GetArrayFromImage(lvwall_thickness)
    np_lvwall_thickness[np.where(np.isnan(np_lvwall_thickness))] = 0
    np_lvwall_thickness[np.where(np_lvwall_thickness > 0)] = 1
    lvwall_mask = utils.np_to_im(np_lvwall_thickness, lvwall_thickness, sitk.sitkUInt8)

    if dilate_wall:
        # Dilate a bit the LV wall to help the projection of aha labels using the probe filter
        dilate = sitk.BinaryDilateImageFilter()
        spacing = lvepi_mask.GetSpacing()
        dilate_voxels = int(np.round(np.divide(dilate_mm, np.array(spacing[0]))))
        print('Dilating wall', dilate_voxels, 'voxels, (', dilate_mm, 'mm)')
        dilate.SetKernelRadius(1)
        dilate.SetKernelRadius(dilate_voxels)
        wall_dilated = dilate.Execute(lvwall_mask)
        np_lvwall_dil = sitk.GetArrayFromImage(wall_dilated)
    np_lvepi = sitk.GetArrayFromImage(lvepi_mask)
    np_lvwall = sitk.GetArrayFromImage(lvwall_mask)
    np_lvendo = sitk.GetArrayFromImage(lvendo_mask)

    # Find LV extension in the z axis
    z_extension_wall = np.unique(np.where(np_lvwall == 1)[0])  # z is [0]
    z_extension_endo = np.unique(np.where(np_lvendo == 1)[0])
    # long_axis_span = np.max(z_extension_wall) - np.min(z_extension_wall)
    # find z-slice index corresponding to the middle of LV wall mask
    mid_lv_long_axis = int(np.round(np.divide(np.max(z_extension_wall) + np.min(z_extension_wall), 2)))
    # print('Middle point', mid_lv_long_axis)

    # Apex
    np_apex = np.zeros(np_lvwall.shape)
    for z_slice in range(mid_lv_long_axis, np_lvwall.shape[0]):    # check only from middle point to avoid region with wall but not endo in the base
        # print('Slice', z_slice)
        wall_slice = np_lvwall[z_slice, :, :]
        endo_slice = np_lvendo[z_slice, :, :]
        if np.where(wall_slice == 1)[0].shape[0] > 0:   # there is wall
            # if np.where(endo_slice == 1)[0].shape[0] == 0:   # there is NO endo, i.e. cavity
            if np.where(endo_slice == 1)[0].shape[0] < 10:   # allow small number of voxels, 0 may be too strict
                # print('Apex slice', z_slice)
                np_apex[z_slice, :, :] = 1

    # check that something was found as apex (i.e. endo mask = 0 & epi mask = 1). Otherwise, set last 5 slices as apex
    if len(np.where(np_apex == 1)[0]) == 0:
        print('No apex found according to endo and epi meshes')
        first_apex_slice = np.max(z_extension_wall) - 5      # manually mark last 5 slices as apex
        np_apex[first_apex_slice: np.max(z_extension_wall), :, :] = 1
    else:
        first_apex_slice = np.min(np.where(np_apex == 1)[0])

    np_apex[np.where(np_lvwall == 0)] = 0  # bg still bg
    

    # ONLY SLICES CONTAINING MYOCARDIUM IN ALL 360º ARE INCLUDED
    np_usable_wall = np.zeros(np_lvwall.shape)

    for z_slice in range(np.min(z_extension_wall), first_apex_slice):
        wall_slice = np_lvwall[z_slice, :, :]
        all_y, all_x = np.where(wall_slice == 1)
        # use slice dependent center, just to check the 360
        slice_center_x = np.round(np.divide(np.max(all_x) + np.min(all_x), 2))
        slice_center_y = np.round(np.divide(np.max(all_y) + np.min(all_y), 2))
        # r, theta = cartesian_to_polar(all_x - center_x, all_y - center_y)
        r, theta = utils.cartesian_to_polar(all_x - slice_center_x, all_y - slice_center_y)

        if utils.check_360(thetas=theta, tolerance=0.10):
            np_usable_wall[z_slice, :, :] = 1

    # Create longitudinal divisions
    min_z_usable_wall = np.min(np.where(np_usable_wall == 1)[0])
    max_z_usable_wall = np.max(np.where(np_usable_wall == 1)[0])
    bin_width = int(np.round(np.divide(max_z_usable_wall - min_z_usable_wall, 3)))
    long_bins = np.arange(min_z_usable_wall, max_z_usable_wall, bin_width)

    np_longitudinal = np.zeros(np_lvwall.shape)
    np_longitudinal[long_bins[0]:long_bins[1], :, :] = 1
    np_longitudinal[long_bins[1]:long_bins[2], :, :] = 2
    np_longitudinal[long_bins[2]:max_z_usable_wall, :, :] = 3
    np_longitudinal[max_z_usable_wall:np_lvwall.shape[0], :, :] = 4  # full z, I will then correct with the LV wall mask

    # Create circunferential division
    np_circunf6 = np.zeros(np_lvwall.shape)
    np_circunf4 = np.zeros(np_lvwall.shape)
    bin_theta_width6 = np.divide(2 * np.pi, 6)    # 60º
    bin_theta_width4 = np.divide(2 * np.pi, 4)    # 90º
    bins_theta6 = np.arange(-np.pi, np.pi + bin_theta_width6, bin_theta_width6)  # [-pi, pi]
    bins_theta4 = np.arange(-np.pi, np.pi + bin_theta_width4, bin_theta_width4) + np.pi/4    # desfase


   # Use only apex as axis center for theta
    x_extension = np.unique(np.where((np_lvepi == 1) & (np_apex == 1))[2])
    y_extension = np.unique(np.where((np_lvepi == 1) & (np_apex == 1))[1])
    center_x = np.round(np.divide(np.max(x_extension) + np.min(x_extension), 2))
    center_y = np.round(np.divide(np.max(y_extension) + np.min(y_extension), 2))
    x_mat = np.array([np.arange(512) - center_x, ] * 512)  # center matrix subtracting center_x
    y_mat = np.array([np.arange(512) - center_y, ] * 512).transpose()
    r, theta = utils.cartesian_to_polar(x_mat, y_mat)

    for z_slice in range(min_z_usable_wall, long_bins[2]):
        pos1 = np.where((theta >= bins_theta6[0]) & (theta <= bins_theta6[1]))
        pos2 = np.where((theta > bins_theta6[1]) & (theta <= bins_theta6[2]))
        pos3 = np.where((theta > bins_theta6[2]) & (theta <= bins_theta6[3]))
        pos4 = np.where((theta > bins_theta6[3]) & (theta <= bins_theta6[4]))
        pos5 = np.where((theta > bins_theta6[4]) & (theta <= bins_theta6[5]))
        pos6 = np.where((theta > bins_theta6[5]) & (theta <= 4))  # I was losing the last one (pi / -pi)
        np_circunf6[z_slice, pos1[0], pos1[1]] = 1
        np_circunf6[z_slice, pos2[0], pos2[1]] = 2
        np_circunf6[z_slice, pos3[0], pos3[1]] = 3
        np_circunf6[z_slice, pos4[0], pos4[1]] = 4
        np_circunf6[z_slice, pos5[0], pos5[1]] = 5
        np_circunf6[z_slice, pos6[0], pos6[1]] = 6

    for z_slice in range(long_bins[2], max_z_usable_wall):
        pos1 = np.where((theta >= bins_theta4[0]) & (theta <= bins_theta4[1]))    # -3pi/4 to -pi/4
        pos2 = np.where((theta > bins_theta4[1]) & (theta <= bins_theta4[2]))     # -pi/4 to pi/4
        pos3 = np.where((theta > bins_theta4[2]) & (theta <= bins_theta4[3]))     # pi/4 to 3*pi/4
        pos4 = np.where((theta > bins_theta4[3]) & (theta <= 4))                  # last one is 3*pi/4 to pi + -pi to -3pi/4
        pos5 = np.where((theta > -np.pi) & (theta < bins_theta4[0]))              # use pos4 and pos5 for this piece
        np_circunf4[z_slice, pos1[0], pos1[1]] = 1
        np_circunf4[z_slice, pos2[0], pos2[1]] = 2
        np_circunf4[z_slice, pos3[0], pos3[1]] = 3
        np_circunf4[z_slice, pos4[0], pos4[1]] = 4     # the last piece includes the sign discontinuity
        np_circunf4[z_slice, pos5[0], pos5[1]] = 4

    # Save only dilated version
    if dilate_wall:
        # compute aha segments also in the dilated LV wall and get mesh with labels
        np_aha_wall_dil = np.zeros(np_lvwall_dil.shape)
        np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 1))] = 2
        np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 2))] = 1
        np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 3))] = 6
        np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 4))] = 5
        np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 5))] = 4
        np_aha_wall_dil[np.where((np_longitudinal == 1) & (np_circunf6 == 6))] = 3
        np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 1))] = 8
        np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 2))] = 7
        np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 3))] = 12
        np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 4))] = 11
        np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 5))] = 10
        np_aha_wall_dil[np.where((np_longitudinal == 2) & (np_circunf6 == 6))] = 9
        np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 1))] = 13
        np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 2))] = 16
        np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 3))] = 15
        np_aha_wall_dil[np.where((np_longitudinal == 3) & (np_circunf4 == 4))] = 14
        np_aha_wall_dil[np.where(np_longitudinal == 4)] = 17
        np_aha_wall_dil[np.where(np_lvwall_dil == 0)] = 0  # bg still bg

        aha_wall_dil = sitk.GetImageFromArray(np_aha_wall_dil)
        aha_wall_dil.CopyInformation(lvwall_mask)

        return aha_wall_dil


def base_parcellation(mesh):
    """
    """
    # get mesh point coordinates
    m_coords = np.zeros([mesh.GetNumberOfPoints(), 3])
    for i in range(mesh.GetNumberOfPoints()):
        m_coords[i, :] = np.array(mesh.GetPoint(i))

    # Find LV extension in the z axis
    # Apex region (segment 17) still the same
    # CAREFULL aha array obtained after probing the image has known errors in the regions limits, better use mesh threshold
    # + connectivity
    apex_region = utils.extractlargestregion(utils.pointthreshold(mesh, 'aha', 17, 17))
    m_coords_apex = np.zeros([apex_region.GetNumberOfPoints(), 3])
    for i in range(apex_region.GetNumberOfPoints()):
        m_coords_apex[i, :] = np.array(apex_region.GetPoint(i))
    limit_z = np.min(m_coords_apex[:, 2])

    z_wall_values = np.unique(m_coords[:, 2])
    extension_z = limit_z - np.min(z_wall_values)

    # Create longitudinal divisions
    nbins = 3
    bin_width = np.divide(extension_z, nbins)
    long_bins = np.arange(np.min(z_wall_values), limit_z, bin_width)
    np_longitudinal = np.zeros(mesh.GetNumberOfPoints())
    np_longitudinal[np.where((m_coords[:, 2] >= long_bins[0]) & (m_coords[:, 2] < long_bins[1]))] = 1
    np_longitudinal[np.where((m_coords[:, 2] >= long_bins[1]) & (m_coords[:, 2] < long_bins[2]))] = 2
    np_longitudinal[np.where((m_coords[:, 2] >= long_bins[2]) & (m_coords[:, 2] < limit_z))] = 3
    np_longitudinal[np.where(m_coords[:, 2] >= limit_z)] = 4

    # I have to include this array to do a pointthreshold later, I will remove it afterwards
    array_longitudinal = numpy_to_vtk(np_longitudinal)
    array_longitudinal.SetName('longitudinal')
    mesh.GetPointData().AddArray(array_longitudinal)

    # Create circunferential division
    np_circunf6 = np.zeros(np_longitudinal.shape)
    np_circunf4 = np.zeros(np_longitudinal.shape)
    bin_theta_width6 = np.divide(2 * np.pi, 6)    # 60'
    bin_theta_width4 = np.divide(2 * np.pi, 4)    # 90'
    bins_theta6 = np.arange(-np.pi, np.pi + bin_theta_width6, bin_theta_width6)     # [-pi, pi]
    bins_theta4 = np.arange(-np.pi, np.pi + bin_theta_width4, bin_theta_width4) + np.pi/4    # phase difference

    # using only 1 reference centering point, the apex
    centroid = utils.get_center_of_mass(utils.pointthreshold(mesh, 'aha', 17, 17))
    r, theta = utils.cartesian_to_polar(m_coords[:, 0] - centroid[0], m_coords[:, 1] - centroid[1])  # use centroid of region with longitudinal = section = {1,2,3,4}
    np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[0]) & (theta <= bins_theta6[1]))] = 1    # section labels start at 1
    np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[1]) & (theta <= bins_theta6[2]))] = 2
    np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[2]) & (theta <= bins_theta6[3]))] = 3
    np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[3]) & (theta <= bins_theta6[4]))] = 4
    np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[4]) & (theta <= bins_theta6[5]))] = 5
    np_circunf6[np.where((np_longitudinal < 3) & (theta >= bins_theta6[5]) & (theta <= 4))] = 6
    np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[0]) & (theta <= bins_theta4[1]))] = 1
    np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[1]) & (theta <= bins_theta4[2]))] = 2
    np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[2]) & (theta <= bins_theta4[3]))] = 3
    np_circunf4[np.where((np_longitudinal == 3) & (theta >= bins_theta4[3]) & (theta <= 4))] = 4   # pi discontinuity, continue in the next line
    np_circunf4[np.where((np_longitudinal == 3) & (theta >= -np.pi) & (theta <= -np.pi + np.pi/4))] = 4


    # define the 17 regions
    np_regions = np.zeros(np_circunf6.shape)
    np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 1))] = 2
    np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 2))] = 1
    np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 3))] = 6
    np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 4))] = 5
    np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 5))] = 4
    np_regions[np.where((np_longitudinal == 1) & (np_circunf6 == 6))] = 3
    np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 1))] = 8
    np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 2))] = 7
    np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 3))] = 12
    np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 4))] = 11
    np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 5))] = 10
    np_regions[np.where((np_longitudinal == 2) & (np_circunf6 == 6))] = 9
    np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 1))] = 13
    np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 2))] = 16
    np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 3))] = 15
    np_regions[np.where((np_longitudinal == 3) & (np_circunf4 == 4))] = 14
    np_regions[np.where(np_longitudinal == 4)] = 17

    array = numpy_to_vtk(np_regions)
    array.SetName('regions')
    mesh.GetPointData().AddArray(array)

    mesh.GetPointData().RemoveArray('longitudinal')
    return mesh
