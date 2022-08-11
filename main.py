import os

import SimpleITK as sitk 
import numpy as np
import pyvista as pv

import automate
import utils
import flattening


ct_img_path = 'src/ct.mha'


output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

timer = {} # Tracing calculation time 

## IMAGE SEGMENTATION #########################################################
# Uncomment if you have access to the segmenation model weights
timer = utils.stopwatch_toggle('Segmentation', trace=timer)
coarse_model_path = 'model/coarse.json'
coarse_model_weights = 'model/coarse.h5'

refine_model_path = 'model/refine.json'
refine_model_weights = 'model/refine.h5'
crop_ct_img, crop_lv_endo_img, crop_lv_wall_img, crop_rv_epi_img = automate.dual_unet_segmentation(
    input_img=sitk.ReadImage(ct_img_path),
    coarse_model_kwargs = dict(model_arch=coarse_model_path, weights=coarse_model_weights,),
    refine_model_kwargs = dict(model_arch=refine_model_path, weights=refine_model_weights,),
    )

sitk.WriteImage(crop_ct_img, os.path.join(output_dir, "crop_ct.mha"),)
sitk.WriteImage(crop_lv_endo_img, os.path.join(output_dir, "crop_lv_endo.mha"),)
sitk.WriteImage(crop_lv_wall_img, os.path.join(output_dir, "crop_lv_wall.mha"),)
sitk.WriteImage(crop_rv_epi_img, os.path.join(output_dir, "crop_rv_epi.mha"), )

timer = utils.stopwatch_toggle('Segmentation', trace=timer)


# # SAX REORIENTATION ###########################################################
timer = utils.stopwatch_toggle('SAX', trace=timer)
# load
crop_lv_endo_img = sitk.ReadImage(os.path.join('src', "crop_lv_endo.mha"),)
crop_lv_wall_img = sitk.ReadImage(os.path.join('src', "crop_lv_wall.mha"),)
crop_rv_epi_img = sitk.ReadImage(os.path.join('src', "crop_rv_epi.mha"), )

print('Running SAX reorientation...')
sax_lv_endo_img, sax_lv_wall_img, sax_rv_epi_img =  automate.sax_orientation(
    lv_endo_img=crop_lv_endo_img,
    lv_wall_img=crop_lv_wall_img,
    rv_epi_img=crop_rv_epi_img,
    )
sitk.WriteImage(sax_lv_endo_img, os.path.join(output_dir, "sax_lv_endo.mha"),)
sitk.WriteImage(sax_lv_wall_img, os.path.join(output_dir, "sax_lv_wall.mha"),)
sitk.WriteImage(sax_rv_epi_img, os.path.join(output_dir, "sax_rv_epi.mha"),)
timer = utils.stopwatch_toggle('SAX', trace=timer)

# # THICKNESS COMPUTATION #####################################################
timer = utils.stopwatch_toggle('Thickness Computation', trace=timer)
print('Run thickness computation...')
sax_lv_thickness_img, sax_lv_wall_depth_img = automate.thickness_calculation(
    lv_wall_img=sax_lv_wall_img,
    lv_endo_img=sax_lv_endo_img,
    )

# sitk.WriteImage(sax_lv_thickness_img, os.path.join(output_dir, "sax_lv_thickness.mha"),)
# sitk.WriteImage(sax_lv_wall_depth_img, os.path.join(output_dir, "sax_wall_depth.mha"),)

timer = utils.stopwatch_toggle('Thickness Computation', trace=timer)

## MESHING ####################################################################
timer = utils.stopwatch_toggle('Meshing & Reformat', trace=timer)
print('Run midwall mesh generation...')
lv_midwall_mesh = automate.midwall_mesh_generation(
    lv_thickness_img=sax_lv_thickness_img,
    lv_wall_depth_img=sax_lv_wall_depth_img,
    lv_endo_img=sax_lv_endo_img,
    lv_wall_img=sax_lv_wall_img,
    cache_dir=output_dir,
    )
lv_midwall_mesh.save(os.path.join(output_dir, 'lv_midwall.vtk'))
timer = utils.stopwatch_toggle('Meshing & Reformat', trace=timer)

## BULLSEYE FLATTENING #########################################################
timer = utils.stopwatch_toggle('Bullseye Computation', trace=timer)
print('Run Bullseye Flattening...')
bullseye_mesh = flattening.get_bullseye_mesh(lv_midwall_mesh,)
pv.wrap(bullseye_mesh).save(os.path.join(output_dir, 'lv_flatten.vtk'))
timer = utils.stopwatch_toggle('Bullseye Computation', trace=timer)

## CVAE Prediction and GradCAMPP ###############################################
# Uncomment the following if you have access to the CVAE weights.
timer = utils.stopwatch_toggle('Arrhythmia Prediction & Attention Map', trace=timer)
print('Run CVAE Prediction...')
cvae_weights = os.path.join('model', 'cvae.h5')
class_prediction, bullseye_mesh, lv_midwall_mesh = automate.cvae_prediction(bullseye_mesh, 
        lv_midwall_mesh=lv_midwall_mesh,
        model_weights=cvae_weights,
        )

# Write output file
bullseye_mesh.save(os.path.join(output_dir, 'lv_flatten_att.vtk'))
lv_midwall_mesh.save(os.path.join(output_dir, 'lv_midwall_att.vtk'))
np.savetxt(os.path.join(output_dir, 'prediction_score.csv'), 
        class_prediction,
        delimiter=',', 
        header='Negative, Positive',
        comments='',
        fmt='%.4f',
        )

print(f"Prediction score: {class_prediction[0]}")
timer = utils.stopwatch_toggle('Arrhythmia Prediction & Attention Map', trace=timer)

# Write Timer
t = np.array(list(timer.values())).round()[None]
h = list(timer.keys())
np.savetxt(os.path.join(output_dir, 'timer.csv'), 
        t, 
        header=','.join(h), 
        delimiter=',', 
        comments='', 
        fmt='%.2f', # write as interger
        )
