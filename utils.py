import tensorflow as tf

def pack_state(mean, covar):
    return tf.concat([mean]+covar, -1)

def unpack_state(state):
    lod = int(state.get_shape().as_list()[-1]/5)\
    mean = state[...,:2*lod]
    covar_upper = state[...,2*lod:3*lod]
    covar_lower = state[...,3*lod:4*lod]
    covar_side = state[...,4*lod:]
    return mean, [covar_upper, covar_lower, covar_side]

def pack_input(obs_mean, obs_covar):
    return tf.concat([obs_mean, obs_covar], axis=-1)

def unpack_input(input):
    lod = int((input.get_shape().as_list()[-1]-1)/2)
    obs_mean = input[...,:lod]
    obs_covar = input[...,lod: -1]
    obs_valid = tf.cast(input[...,-1], tf.bool)
    return obs_mean, obs_covar
