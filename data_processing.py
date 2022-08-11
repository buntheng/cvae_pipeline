import itertools
import functools
import SimpleITK as sitk
import numpy as np
import tensorflow as tf

import flattening


def preprocessing_cvae_input(input_mesh, 
        input_shape,
        clip=10,
        as_tensor=False,
        ):
    """
    """
    thickness_np, mask_np = flattening.flat_mesh_to_np(input_mesh, 
            output_shape=input_shape[0],
            scalar_name='thickness',
            return_mask=True,
            )
    thickness_np[thickness_np > clip] = clip
    thickness_np = thickness_np/ clip
    if as_tensor:
        thickness_np = tf.convert_to_tensor(thickness_np)
        mask_np = tf.convert_to_tensor(mask_np)
    return thickness_np[None, ..., None], mask_np[None, ..., None]


def preprocessing_segment_input(input_img, 
        input_shape,):
    """
    """
    img_resize = resize(input_img, 
                        new_size=input_shape, 
                        interpolation_fn='sitkLinear',
                        ) 
    input_np = sitk.GetArrayFromImage(img_resize)
    return input_np[None, ..., None]


def resize(img: sitk.Image,
           new_size=[128, 128, 128],
           interpolation_fn="sitkLinear",
        ):
    """ 
    """
    img_dim = img.GetDimension()
    orig_size = np.array(img.GetSize())
    orig_spacing = np.array(img.GetSpacing())

    ref_size = np.array(new_size).astype(int)
    phys_size = orig_spacing * orig_size
    ref_spacing = phys_size/ref_size
    ref_img = sitk.Image(ref_size.astype(np.uint).tolist(),
                         img.GetPixelIDValue())
    ref_img.SetDirection(img.GetDirection())
    ref_img.SetOrigin(img.GetOrigin())
    ref_img.SetSpacing(ref_spacing)

    # Resample new image using identity tranformation
    id_transform = sitk.Transform()
    id_transform.SetIdentity()

    interpolation_fn = getattr(sitk, interpolation_fn)
    img = sitk.Resample(img, ref_img, id_transform, interpolation_fn)
    return img


def revert_resize(img, 
                ref_img=None,
                interpolation_fn="sitkLinear",
                pixel_id_value=None,
                ):
    """ Revert image back to reference physical attribute.
    .. TODO: creating the algorithm for non homogenous current image spacing.
    """
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    img_dim = img.GetDimension()

    # correct resize Spacing
    size = ref_img.GetSize()
    spacing = ref_img.GetSpacing()
    direction = ref_img.GetDirection()
    origin = ref_img.GetOrigin()

    # Get Current spacing from 
    phys_size = np.multiply(size, spacing)
    new_spacing = np.divide(phys_size, orig_size)
    img.SetSpacing(new_spacing.tolist())
    
    # correct Origin and Direction
    img.SetOrigin(origin)
    img.SetDirection(direction)

    id_transform = sitk.Transform()
    id_transform.SetIdentity()
    if pixel_id_value is not None:
        ref_img = sitk.Cast(ref_img, pixel_id_value)
    
    interpolation_fn = getattr(sitk, interpolation_fn)
    new_img = sitk.Resample(img, ref_img, id_transform, interpolation_fn)
    new_img.CopyInformation(ref_img) # copy metadata
    return new_img


def postprocessing_segment_output(
        array,
        ref_img, 
        ):
    """
    """
    imgs = []
    # apply value threshold
    threshold_val = 0.5
    array = (array > threshold_val).astype(np.int8)

    for chn in range(array.shape[-1]):
        _array = array[..., chn]
        _img = sitk.GetImageFromArray(_array)
        # resample to original geometry
        _img = revert_resize(img=_img, 
                ref_img=ref_img,
                interpolation_fn="sitkNearestNeighbor")

        # get biggest connected pixel cluster
        _img = biggest_cluster(_img)
        imgs.append(_img)
    # remove overlap
    return remove_overlap(*imgs)


def biggest_cluster(mask,):
    """ 
    :param mask: binary mask. 
    """
    mask = sitk.Cast(mask, sitk.sitkInt8)

    mask = sitk.ConnectedComponent(mask)
    #mask = sitk.Cast(mask, sitk.sitkInt8)

    labelfilter = sitk.LabelStatisticsImageFilter()
    labelfilter.Execute(mask, mask)
    
    labels = labelfilter.GetLabels()

    count = -np.inf
    largest_label = 0
    for label in labels:
        if label != 0:
            label_count = labelfilter.GetCount(label)
            if label_count > count:
                count = label_count
                largest_label = label

    mask = sitk.BinaryThreshold(mask, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)
    return mask


def remove_overlap(*args):
    """ Early images have higher priority.
    TODO: dbg
    """
    new_args = list(args)
    for ind in range(1, len(args)):
        new_args[ind] = sitk.Xor(new_args[ind], sitk.And(new_args[ind], new_args[ind-1]))
    return new_args 


def bbox_crop_img(input_img, 
        binary_imgs,
        padding=0,
        ):
    """
    """
    if not isinstance(input_img, (list, tuple)):
        input_img = [input_img]
    # get bbox
    binary_img = functools.reduce(sitk.Or, binary_imgs)
    bbox = get_bbox(binary_img, padding=padding)

    output_imgs = []
    for _img in input_img:
        _img = crop_from_bbox(_img, bbox)    
        if len(input_img) == 1:
            return _img
        output_imgs.append(_img)
    return  tuple(output_imgs)


def crop_from_bbox(img:sitk.Image,
                   bbox:list,):
    """
    """
    # get cropping indices
    min_ = [int(bbox[2*i]) for i in range(3)]
    max_ = [int(bbox[2*i+1]+1) for i in range(3)] # bbox are stored as (x_min, x_max, ...)
                                                    # + 1 so slicing would take into account the max pixel
    return img[min_[0]:max_[0], min_[1]:max_[1], min_[2]:max_[2]]


def get_bbox(img, 
    label_val=1, 
    padding=None,
    ):
    """
    :param padding: apply lower and upper padding 
    :type padding: int/list(int)
    """

    labelfilter = sitk.LabelStatisticsImageFilter()
    labelfilter.Execute(img, img)
    bbox = labelfilter.GetBoundingBox(label_val)

    min_ = [int(bbox[2*i]) for i in range(3)]
    max_ = [int(bbox[2*i+1]+1) for i in range(3)] # bbox are stored as (x_min, x_max, ...)

    if padding is not None: 

        if not isinstance(padding, (list, tuple)):
            padding = [padding] * 3
        
        img_size = img.GetSize()
        min_ = [min_[i]-padding[i] if min_[i]-padding[i] >= 0 else 0 for i in range(3)]
        max_ = [max_[i]+padding[i] if max_[i]+padding[i] <= img_size[i] else img_size[i] for i in range(3)]
        
    bbox = list(itertools.chain(*zip(min_, max_))) # repackage bbox
    return bbox
