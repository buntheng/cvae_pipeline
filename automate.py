import os
import subprocess
import shlex

import SimpleITK as sitk
import numpy as np
import pyvista as pv

from pyezzi import ThicknessSolver

import model
import data_processing
import utils
import orientation
import parcellation
import flattening
import gradcampp


def dual_unet_segmentation(input_img,
        coarse_model_kwargs,
        refine_model_kwargs, # keep as None for coarse segmentation
        ):
    """
    """
    # load models
    coarse_model = model.load_model(**coarse_model_kwargs)
    input_np = data_processing.preprocessing_segment_input(input_img=input_img,
            input_shape=coarse_model.inputs[0].shape[1:-1].as_list(), # [1:-1] remove batch and channel from shape
            )
    pred_np = coarse_model(input_np, training=False,).numpy()[0] # [0]: get the first batch

    # revert segmentation to original geometry 
    lv_endo_img, lv_wall_img, rv_epi_img = data_processing.postprocessing_segment_output(
            array=pred_np,
            ref_img=input_img, # using original image as reference
            )

    refine_model = model.load_model(**refine_model_kwargs)

    # ROI cropping
    crop_input_img, crop_rv_epi_img = data_processing.bbox_crop_img(
            input_img=[input_img, rv_epi_img],
            binary_imgs=[lv_endo_img, lv_wall_img, rv_epi_img],
            )

    # preprocessing segmentation input
    crop_input_np = data_processing.preprocessing_segment_input(
            input_img=crop_input_img, 
            input_shape=refine_model.inputs[0].shape[1:-1].as_list(),
            )
    # run segmentation
    refine_pred_np = refine_model(crop_input_np, training=False).numpy()[0]

    # run post-processing
    crop_lv_endo_img, crop_lv_wall_img = data_processing.postprocessing_segment_output(
        array=refine_pred_np,
        ref_img=crop_input_img, # using original image as reference
        )

    # remove overlap between lvwall and rvepi
    crop_lv_wall_img, crop_rv_epi_img = data_processing.remove_overlap(crop_lv_wall_img, crop_rv_epi_img)
    return crop_input_img, crop_lv_endo_img, crop_lv_wall_img, crop_rv_epi_img


def sax_orientation(
        # ct_img,
        lv_endo_img,
        lv_wall_img,
        rv_epi_img,
        ):
    """ Run 
    """
    output_imgs = orientation.orientation(
            input_imgs = dict(
                    # intensity=ct_img,
                    lvendo=lv_endo_img,
                    lvwall=lv_wall_img,
                    rvepi=rv_epi_img,
                    ),
            )
    return output_imgs['lvendo'], output_imgs['lvwall'], output_imgs['rvepi']


def thickness_calculation(lv_wall_img, 
        lv_endo_img,
        ):
    """ Calculating LV wall thickness.
    """
    lv_wall_np = sitk.GetArrayFromImage(lv_wall_img)
    lv_endo_np = sitk.GetArrayFromImage(lv_endo_img)
    label_np = 2*lv_wall_np + lv_endo_np
    label_np[label_np > 2] = 2
    solver = ThicknessSolver(
            labeled_image=label_np, 
            spacing=lv_wall_img.GetSpacing()[::-1],
            )
    thickness = solver.result
    thickness[np.isnan(thickness)] = 0
    L0 = solver.L0
    L0[np.isnan(thickness)] = 0
    wall_depth = L0 / thickness
    
    thickness_img = sitk.GetImageFromArray(thickness)
    wall_depth_img = sitk.GetImageFromArray(wall_depth)
    
    ref_img = sitk.Cast(lv_wall_img, sitk.sitkFloat32)
    thickness_img.CopyInformation(ref_img)
    wall_depth_img.CopyInformation(ref_img)
    return thickness_img, wall_depth_img
    

def midwall_mesh_generation(
        lv_thickness_img,
        lv_wall_depth_img,
        lv_endo_img,
        lv_wall_img,
        cache_dir,
        ):
    """
    """
    ## Get Mid-wall surface mesh 
    wall_depth_np = sitk.GetArrayFromImage(lv_wall_depth_img)
    endo_np = sitk.GetArrayFromImage(lv_endo_img)
    thres_depth_np = (wall_depth_np < 0.5).astype(int) + endo_np
    thres_depth_img = sitk.GetImageFromArray(thres_depth_np)
    thres_depth_img.CopyInformation(lv_wall_depth_img)

    midwall_mesh = utils.marching_cubes(thres_depth_img, 
            closing_radius=5,
            n_clusters=30000,
            )
    # endo_mesh = utils.marching_cubes(lv_endo_img, 
    #         closing_radius=5, 
    #         n_clusters=30000,)
    # remove mitral_valve
    # midwall_mesh = midwall_mesh.clip_surface(endo_mesh, invert=True)
    lv_thickness_np = sitk.GetArrayFromImage(lv_thickness_img)
    lv_thickness_np[np.isnan(lv_thickness_np)] = 50
    lv_thickness_np *= 1000
    _lv_thickness_img = sitk.GetImageFromArray(lv_thickness_np)
    _lv_thickness_img.CopyInformation(lv_thickness_img)
    thickness_mesh = utils.probe_img_2_mesh(mesh=midwall_mesh, img=_lv_thickness_img)
    midwall_mesh['thickness'] = thickness_mesh['MetaImage'] / 1000

    # get lv parcellation
    lv_wall_aha = parcellation.lvwall_aha_parcellation(
            lvwall_thickness=lv_thickness_img, 
            lvendo_mask=lv_endo_img,
            lvepi_mask=sitk.Or(lv_endo_img, lv_wall_img),
            )
    aha_mesh = utils.probe_img_2_mesh(midwall_mesh, lv_wall_aha)
    midwall_mesh['aha'] = aha_mesh['MetaImage']

    midwall_mesh = parcellation.base_parcellation(midwall_mesh)
    midwall_mesh, _ = flattening.split_base(midwall_mesh)
    return midwall_mesh


def cvae_prediction(bullseye_mesh,
        lv_midwall_mesh,
        model_weights=None,
        ):
    """
    """
    models = model.cvae()
    if model_weights is not None:
        models['main'].load_weights(model_weights)

    thickness_np, mask_np = data_processing.preprocessing_cvae_input(bullseye_mesh,
            input_shape=models['main'].inputs[0].shape[1:-1].as_list(),
            as_tensor=True,
            )

    (neg_att_map, pos_att_map), prediction = gradcampp.prediction(batch_input = [thickness_np, mask_np], 
            models=models,)
    
    # remap to mesh
    for arr, nm in zip([neg_att_map, pos_att_map], ["att_neg", "att_pos"]):
        bullseye_mesh, lv_midwall_mesh = flattening.np_to_mesh(array=arr[0], # batch size == 1
                array_name=nm,
                disk_mesh=bullseye_mesh,
                surface_mesh=lv_midwall_mesh,)

    return prediction, bullseye_mesh, lv_midwall_mesh
    

