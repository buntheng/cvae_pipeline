# import re
# import os
import pdb
import functools
import operator

# from tqdm import tqdm 
# import imageio as io
import numpy as np
import skimage.transform as skt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def get_m_pp(derivs, F):
    """
    """
    first_d, second_d, third_d = derivs
    sum_A_ab = tf.reduce_sum(F, axis=(1,2)) # get global sum in a, b  axis
    # weights = tf.nn.relu(first_d)
    weights = first_d

    alpha_nom = second_d
    alpha_denom = 2*second_d + sum_A_ab * third_d
    # remove 0 from alpha_denom 
    alpha_denom = tf.tensor_scatter_nd_update(alpha_denom, 
            indices=tf.where(alpha_denom==0), 
            updates=tf.ones(len(tf.where(alpha_denom==0))))
    alpha = alpha_nom/alpha_denom

    # normalized alpha 
    alpha_thresholding = tf.where(weights!=0, alpha, 0.0) # replace weights==0 with 0.0
    norm_constant = tf.reduce_sum(alpha_thresholding, axis=(1,2))
    # remove 0
    norm_constant_processed = tf.where(norm_constant!=0, 
            norm_constant, 
            1.)
    alpha /= tf.reshape(norm_constant_processed, (-1, 1, 1, first_d.shape[-1]))

    deep_linear_weights = tf.reduce_sum(weights * alpha, axis=(1, 2))
    deep_linear_weights = tf.nn.relu(deep_linear_weights)

    # deep_linear_weights = tf.reduce_sum(alpha, axis=(1,2))
    # weights = tf.nn.relu(first_d)
    grad_CAM_map = tf.reduce_sum(deep_linear_weights*F, axis=-1)
    cam = tf.nn.relu(grad_CAM_map)
    # cam = grad_CAM_map

    # rescale 
    cam /= tf.reduce_max(cam)
    return cam


def get_m(derivs, F):
    first_d, _, _ = derivs 
    weights = tf.reduce_mean(first_d, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(F*weights, axis=-1)
    return cam


def get_derivs_exp(cost, gradient, index):
    """
    """
    first_d = tf.exp(cost)[..., index]*gradient
    second_d = tf.exp(cost)[..., index]*gradient*gradient
    third_d = tf.exp(cost)[..., index]*gradient*gradient*gradient
    return [first_d.numpy(), second_d.numpy(), third_d.numpy()]


def get_derivs(costs, gradients):
    """
    """
    first_ds = []
    costxgrad_sum = functools.reduce(operator.add, 
            [costs[ind][..., ind] * gradients[ind] 
                for ind in range(len(gradients))]
            )
    for ind, (cost, grad) in enumerate(zip(costs, gradients)):
        _d = cost[..., ind] * (grad - costxgrad_sum)
        first_ds.append(_d.numpy())

    dxgrad_sum = functools.reduce(operator.add, 
            [first_d * grad
                for first_d, grad  in zip(first_ds, gradients)]
            )
    second_ds = []
    for ind, (cost, grad, first_d) in enumerate(zip(costs, gradients, first_ds)):
        _d = first_d * (grad - costxgrad_sum) - cost[..., ind] * dxgrad_sum
        second_ds.append(_d.numpy())

    d2xgrad_sum = functools.reduce(operator.add, 
            [ second_d * grad
                for second_d, grad in zip(second_ds, gradients)]
            )
    third_ds = []
    for ind, (cost, grad, first_d, second_d) in enumerate(zip(costs, gradients, first_ds, second_ds)):
        _d = second_d*(grad - costxgrad_sum) - 2*first_d*dxgrad_sum - cost[..., ind] * d2xgrad_sum
        third_ds.append(_d.numpy())

    return [(d1, d2, d3) for d1, d2, d3 in zip(first_ds, second_ds, third_ds)]


def prediction(batch_input,
        models,
        auto_zoom=True,
        deepest_key=True,
        ):
    """
    """
    main_model = models['main']
    main_model.trainable = True # for gradient calculation
    encoder = models['encoder']
    classifying_model = models['classifier_model']

    shape = batch_input[0].shape[1:-1].as_list()
   
    z_dict = {}
    class_output_dict = {}
    conv_output = {}
    with tf.GradientTape(persistent=True) as tape: 
        tape.watch(batch_input)
        concat = encoder.layers[2]([batch_input[0], batch_input[1]])
        x = concat
        for layer in encoder.layers[3:-3]:
            x = layer(x)
            if 'conv' in layer.name:
                conv_output[layer.name] = x
                tape.watch(conv_output[layer.name])
                _F = x # trace last conv layer
        for layer in encoder.layers[-3:]:
            name = layer.name
            if name == 'mean':
                mean = layer(x)
                tape.watch(mean)

            elif name == 'log_var':
                log_var = layer(x)

            elif name == 'gaussian_sampling':
                z = layer([mean, log_var])

        # classification
        x = mean 
        for _layer in classifying_model.layers: 
            x = _layer(x)
        class_output = x

        zs = [z[..., i] for i in range(z.shape[-1])]

        class_outputs = [class_output*tf.one_hot(i, 2) for i in range(class_output.shape[-1])]

    # if deepest_key:
    att_map = []
    key = list(conv_output.keys())[-1]
    gradients = [tape.gradient(_output, _F).numpy() for _output in class_outputs]
    derivs = get_derivs(class_outputs, gradients)
    m = [get_m_pp(_derivs, _F) for _derivs in derivs]
    if auto_zoom:
        m = [skt.resize(_m, [_m.shape[0]] + shape) for _m in m]
    att_map = m

    # else: 
    #     att_map = {}
    #     for _ind, key in enumerate(conv_output.keys()):
    #         _F = conv_output[key]
    #         ## get y
    #         gradients = [tape.gradient(_output, _F).numpy() for _output in class_outputs]
    #         derivs = get_derivs(class_outputs, gradients)
    #         m = [get_m_pp(_derivs, _F) for _derivs in derivs]
    #         if auto_zoom:
    #             m = [skt.resize(_m, (_m.shape[0], output_shape, output_shape)) for _m in m]

    #         att_map[key] = m

    # decoder = models['decoder']
    # recon = decoder.predict([x, batch_input[1]])
    return att_map, class_output # , recon

