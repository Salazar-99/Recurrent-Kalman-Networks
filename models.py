import numpy as np 
import tensorflow as tf

#Recurrent Kalman Network model
class RKN(tf.keras.Model):
    def __init__(self, observation_shape, latent_observation_dim, output_dim, num_basis, bandwidth, 
                trans_net_hidden_units=[], never_invalid=False, cell_type='rkn')
        #Inherit standard functionality from tf.keras.Model
        super().__init__()
        self._obs_shape = observation_shape
        self._lod = latent_observation_dim
        #Latent state dimension
        self._lsd = 2 * self._lod
        self._output_dim = output_dim
        self._never_invalid = never_invalid
        #Flag for low-dimensional output
        self._ld_output = np.isscalar(self._output_dim)

        #build_encoder_hidden() is problem specific and is implemented in subclasses of RKN
        self._enc_hidden_layers = self._time_distributee_layers(self.build_encoder_hidden())

        #N(0,0.05) initialization to prevent NaN in normalization step
        self._layer_w_mean = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self._lod, activation='linear', bias_initializer=tf.random_normal_initializer()))
        #Normalization layer
        self._layer_w_mean_norm = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: x / tf.norm(x, ord='euclidean', axis=-1, keepdims=True)))
        #Covariance has elu(x)+1 activation to ensure positive values
        self._layer_w_covars = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self._lod, activation=lambda x: tf.keras.activations.elu(x)+1))

    def call(self, inputs):
        