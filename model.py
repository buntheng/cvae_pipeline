import functools
import operator

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa 


def load_model(model_arch:str, 
        weights:str=None,
        ):
    """
    """
    with open(model_arch, "r") as F:
        model = keras.models.model_from_json(F.read(), 
                custom_objects={"InstanceNormalization": tfa.layers.InstanceNormalization})
    if weights is not None:
        model.load_weights(weights)
        print(f"Load weights from: {weights}")
    return model


def gaussian_sampling(args):
    """
    """
    mean, log_var = args
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1:]
    epsilon = K.random_normal(shape=dim)
    return mean + K.exp(0.5 * log_var) * epsilon


def encoding2d(inputs, 
        filters=[8, 16, 32]): 
    """ swol 3D in, smol 3D out
    """
    if isinstance(inputs, (list, tuple)) and len(inputs) > 1: 
        x = keras.layers.Concatenate(name="en_concat")(inputs)

    else: 
        x = inputs
    # apply compress to latent information
    for ind, _filter in enumerate(filters): 
        x = keras.layers.Conv2D(
                filters=_filter,
                kernel_size=3,
                strides=1,
                activation="relu",
                name=f"en_conv_{ind}_i",
                padding="same",
                )(x)
        x = keras.layers.Conv2D(
                filters=_filter,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu", name=f"en_conv_{ind}_ii",)(x)
    return x


def decoding2d(inputs, 
        output_channel=1,
        filters:list=None,
        double_stacked=False,
        ): 
    """ smol 3D in, swol 3D out
    """
    if isinstance(inputs, (list, tuple)):
        x, cD = inputs

    else:
        x = inputs
        cD = None

    cD_strides = 2**len(filters)
    for ind, _filter in enumerate(filters): 
        # concatenate 
        if cD is not None: 
            _cD = keras.layers.Conv2D(
                    kernel_size=3, 
                    filters=_filter, 
                    strides=cD_strides, 
                    activation="relu",
                    name=f"de_cond_{ind}_i",
                    padding="same",
                    )(cD)
            if double_stacked:
                _cD = keras.layers.Conv2D(
                        kernel_size=3, 
                        filters=_filter, 
                        strides=1, 
                        activation="relu",
                        name=f"de_cond_{ind}_ii",
                        padding="same",
                        )(_cD)
            cD_strides //= 2
            # concatenate conditioned with main decoder output
            x = keras.layers.Concatenate(name=f"de_conc_{ind}")([x,  _cD])

        x = keras.layers.Conv2DTranspose(
                kernel_size=3, 
                filters=_filter, 
                strides=2, 
                activation="relu",
                name=f"de_convt_{ind}_i",
                padding="same",
                )(x)
        if double_stacked:
            x = keras.layers.Conv2D(
                    kernel_size=3, 
                    filters=_filter, 
                    strides=1, 
                    activation="relu",
                    padding="same",
                    name=f"de_convt_{ind}_ii",
                    )(x)
    x = keras.layers.Conv2D(
                kernel_size=3, 
                filters=output_channel, 
                padding="same",
                strides=1, 
                activation="relu",
                name="de_convt_out",
                )(x)
    return x


def classifying(inputs, 
        classify_mode=None,
        filters=[16, 8, 4], 
        output_channel=1): 
    """ vector 3D in vector 3D out.
    """
    x = inputs
    for ind, _filter in enumerate(filters): 
        x = keras.layers.Dense(units=_filter, 
                # activation_fn='tanh',
                name=f"cl_dense_{ind}_i")(x)
        x = keras.layers.Dense(units=_filter, 
                activation='tanh',
                name=f"cl_dense_{ind}_ii")(x)

    _x = keras.layers.Dense(units=output_channel, 
            name='cl_output')(x)
    x = keras.activations.softmax(_x)
    return x, _x


def cvae(
        input_shape=[256, 256],
        filters=[8, 16, 32, 64],
        latent_dim=25,
        bottleneck=16,
        *args, **kwargs,
    ): 
    """
    """
    input_shape += [1] # add channel dimension
    mask_input = keras.layers.Input(shape=input_shape, name="mask")
    thickness_input = keras.layers.Input(shape=input_shape, name="thickness")

    ##  ENCODER
    encoder_output = encoding2d(inputs=[thickness_input, mask_input], 
            filters=filters)
    encoding_shape = K.int_shape(encoder_output)

    x = keras.layers.Flatten(name="mid_flat")(encoder_output)
    x = keras.layers.Dense(units=bottleneck)(x) # add bottleneck
    
    mean = keras.layers.Dense(units=latent_dim, name="mean")(x)
    log_var = keras.layers.Dense(units=latent_dim, name="log_var")(x)
    z = keras.layers.Lambda(gaussian_sampling, name="gaussian_sampling")([mean, log_var])
    encoder_model = keras.Model(inputs=[thickness_input, mask_input], outputs=[mean, log_var, z], name="encoder_model")

    # CLASSIFIER
    classifying_input = keras.layers.Input(shape=z.shape[1:], name='class_img_info')
    class_output_channel = 2

    classify_output, _classify_output = classifying(inputs=classifying_input,
            output_channel=class_output_channel, 
            # classify_mode=classify_mode,
            )
    classifying_model = keras.Model(inputs=classifying_input, 
            outputs=classify_output, 
            name="classify_model",)

    # DECODER
    decoder_input = keras.layers.Input(shape=z.shape[-1], name='decoder_input')
    # reshape  vector to 4d + channel tensor
    x = keras.layers.Dense(units=functools.reduce(operator.mul, encoding_shape[1:]))(decoder_input)
    x = keras.layers.Reshape(target_shape=encoding_shape[1:], name="3d_reshape")(x)
    decoder_output = decoding2d(inputs=[x, mask_input], 
            filters=filters[::-1], 
            output_channel=1) # reverse encoding order
    decoder_output *= mask_input

    decoder_model = keras.Model(inputs=[decoder_input, mask_input], outputs=decoder_output, name="decoder_model")
    
    # MAIN MODEL
    encoder_output = encoder_model([thickness_input, mask_input])
    
    # used mean as classify input
    classify_output_z = classifying_model(encoder_output[-1])
    classify_output_mean = classifying_model(encoder_output[0])
    classify_output = [classify_output_z, classify_output_mean]
    decoder_mean_output = decoder_model([encoder_output[0], mask_input])

    # Get decoder output from z (to train the VAE)
    decoder_output = decoder_model([encoder_output[-1], mask_input])

    main_model = keras.Model(inputs=[thickness_input, mask_input], 
            outputs=[decoder_output, decoder_mean_output, encoder_output, classify_output], name="main_model")

    return {
            "main": main_model, 
            "encoder": encoder_model, 
            "decoder": decoder_model,
            "classifier_model": classifying_model,
            }
