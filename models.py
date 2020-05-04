import numpy as np 
import tensorflow as tf
import math as m
from utils import pack_input, unpack_state

#Defining pi as a TF constant for performance
pi = tf.constant(m.pi)

#Recurrent Kalman Network model
class RKN(tf.keras.Model):
    #Constructor
    def __init__(self, observation_shape, latent_observation_dim, output_dim, num_basis, bandwidth, 
                trans_net_hidden_units=[], cell_type='rkn')
        #Inherit standard functionality from tf.keras.Model
        super().__init__()
        self._obs_shape = observation_shape
        self._lod = latent_observation_dim
        #Latent state dimension
        self._lsd = 2 * self._lod
        self._output_dim = output_dim
        #Flag for low-dimensional output
        self._ld_output = np.isscalar(self._output_dim)

        #The build_encoder_hidden() function is problem specific and is implemented in subclasses of RKN
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
        
        #RKN Transition cell
        self._cell = RKNTransitionCell(self._lsd, self._lod, number_of_nasis=num_basis, bandwidth=bandwidth,
                                        trans_net_hidden_units=trans_net_hidden_units, never_invalid=never_invalid)

        #RNN layer with RKN cell
        self._layer_rkn = tf.keras.layers.RNN(self._cell, return_sequences=True)
        
        #Decoder (low-dimensional)
        if self._ld_output:
            #Decoder mean
            self._layer_dec_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self._output_dim))
            #Decoder variance
            self._var_dec_hidden = self._time_distribute_layers(self.build_var_decoder_hidden())
            self._layer_var_dec_out = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(units=self._output_dim, activation=lambda x: tf.keras.activations.elu(x)+1))
        
        #Decoder (high-dimensional)
        else:
            self._layer_dec_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2dTranspose(
                                        self._output_dim[-1], kernel_size=3, padding='same', activation='sigmoid'))

        #Creating input shape with 'None' as the sequence length (only allowing lists as observation shape)
        if isinstance(observation_shape, list):
            in_shape = [None] + observation_shape
        else:
            raise AssertionError('Observation shape must be a list')
                                                            
        #Creating Input Layer
        inputs = tf.keras.layers.Input(shape=in_shape)
        
    #Implements forward pass        
    def call(self, inputs):
        #Encoder
        enc_last_hidden = self._prop_through_layers(inputs, self._enc_hidden_layers)
        w_mean = self._layer_w_mean_norm(self._layer_w_mean(enc_last_hidden))
        w_covar = self._layer_w_covar(enc_last_hidden)
        #Transition
        rkn_in = pack_input(w_mean, w_covar)
        z = self._layer_rkn(rkn_in)
        post_mean, post_covar = unpack_state(z)
        post_covar = tf.concat(post_covar, -1)
        #Decoder
        pred_mean = self._layer_dec_out(self._prop_through_layers(post_mean, self._dec_hidden))
        if self._ld_output:
            pred_var = self._layer_var_dec_out(self._prop_through_layers(post_mean, self._dec_hidden))
            return tf.concat([pred_mean, pred_var], -1)
        else:
            return pred_mean
   
    #Loss functions
    def gaussian_neg_log_likelihood(self, targets, pred_mean_var):
        pred_mean, pred_var = pred_mean_var[,:self._output_dim], pred_mean_var[,self.output_dim:]
        pred_var += 1e-8
        element_wise_nll = 0.5*(tf.log(2*pi)+tf.log(pred_var)+((target-pred_mean)**2/pred_var)
        sample_error = tf.reduce_sum(element_wise_nll, axis=-1)
        return tf.reduce_mean(sample_error)

    def rmse(self, target, pred_mean_var):
        pred_mean = pred_mean_var[,:self._output_dim]
        return tf.sqrt(tf.reduce_mean((pred_mean-target)**2))

    def bernoulli_neg_log_likelihood(self, targets, predictions, uint8_targets=True):
        #Normalizing integers (0-255) to floats (0-1)
        if unint8_targets:
            targets = targets/255.0
        point_wise_error = -(targets*tf.log(predictions+1e-12)+(1-targets)*tf.log(1-predictions+1e-12))
        #TODO: Not sure why this is required
        red_axis = [i+2 for i in range(len(targets.get_shape())-2)]
        sample_error = tf.reduce_sum(point_wise_error, axis=red_axis)
        return tf.reduce_mean(sample_error)

    #Helper functions for call
    def _prop_through_layers(inputs, layers):
        h = inputs
            for layer in layers:
                h = layer(h)
        return h

    def _time_distribute_layers(layers):
        td_layers = []
        for l in layers:
            td_layers.append(tf.keras.layers.TimeDistributed(l))
        return td_layers
    
    #Builder functions to be implemented in subclasses
    def build_encoder_hidden(self):
        raise NotImplementedError

    def build_decoder_hidden(self):
        raise NotImplementedError

    def build_var_decoder_hidden(self):
        raise NotImplementedError

#Dense network for use in RKNTransitionCell
class TransitionNet(tf.keras.Model):
    #Constructor 
    def __init__(self, lsd, numbers_of_basis, hidden_units):
        self._hidden_layers = []
        #Set input shape to latent state dimension
        cur_in_shape = lsd
        #Hidden units
        for u in hidden_units:
            layer = tf.keras.layers.Dense(u, activation='relu')
            #TODO: I think this is just specifying input shape, not sure if it's necessary
            layer.build([None, cur_in_shape])
            #Updating input shape for next layer
            cur_in_shape = u
            self._hidden_layers.append(layer)
        #Output layer
        self._out_layer = tf.keras.layers.Dense(number_of_basis, activation='softmax')
        self._out_layer.build([None, cur_in_shape])
    
    #Forward pass
    def call(self, latent_state):
        h = latent_state
        for hidden_layer in self._hidden_layers:
            h = hidden_layer(h)
        return self._out_layer(h)

#Performs Kalman update, implemented as a cell ro be used in an RNN inside RKN
class RKNTransitionCell(tf.keras.layers.Layer):
    #Constructor
    def __init__(self, latent_state_dim, latent_obs_dim, number_of_basis, bandwidth, trans_net_hidden_units=[], initial_trans_covar=0.1):
        super().__init__()
        #n=2*m
        assert latent_state_dim == 2*latent_obs_dim
        self._lsd = latent_state_dim
        self._lod = latent_obs_dim
        self._num_basis = number_of_basis
        self._bandwidth = bandwidth
        self._trans_net_hidden_units = trans_net_hidden_units
        self._initial_trans_covar = initial_trans_covar

    #Forward pass (Prediction and update)
    def call(self, inputs, states, **kwargs)
        #Unpack inputs
        obs_mean, obs_covar = unpack_input(inputs)
        state_mean, state_covar = unpack_state(states[0])
        #Prediction step
        pred_res = self._predict(state_mean, state_covar)
        prior_mean, prior_covar = pred_res
        #Update step
        update_res = self._update(prior_mean, prior_covar, obs_mean, obs_covar)
        #Pack outputs
        post_state = pack_state(post_mean, post_covar)
        #TODO: Not sure if both are needed
        return post_state, post_state

    #Constructor for objects dependant on initial constructor
    def build(self, input_shape):
        #Basis matrices
        tm_11_init =        np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_12_init =  0.2 * np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_21_init = -0.2 * np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_22_init =        np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_11_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_11_basis",
                                     initializer=k.initializers.Constant(tm_11_init))
        tm_12_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_12_basis",
                                     initializer=k.initializers.Constant(tm_12_init))
        tm_21_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_21_basis",
                                     initializer=k.initializers.Constant(tm_21_init))
        tm_22_full = self.add_weight(shape=[self._num_basis, self._lod, self._lod], name="tm_22_basis",
                                     initializer=k.initializers.Constant(tm_22_init))        
        tm_11, tm_12, tm_21, tm_22 = (tf.matrix_band_part(x, self._bandwidth, self._bandwidth) for x in
                                      [tm_11_full, tm_12_full, tm_21_full, tm_22_full])
        self._basis_matrices = tf.concat([tf.concat([tm_11, tm_12], -1),
                                          tf.concat([tm_21, tm_22], -1)], -2)
        self._basis_matrices = tf.expand_dims(self._basis_matrices, 0)

        #Coefficient network
        self._coefficient_net = TransitionNet(self._lsd, self._num_basis, self._trans_net_hidden_units)
        self._trainable_weights += self._coefficient_net.weights

        #Transition covariance
        elup1_inv = lambda x: (np.log(x) if x < 1.0 else (x - 1.0))
        log_transition_covar = self.add_weight(shape=[1, self._lsd], name="log_transition_covar",
                                               initializer=k.initializers.Constant(elup1_inv(self._initial_trans_covar)))
        trans_covar = elup1(log_transition_covar)
        self._trans_covar_upper = trans_covar[:, :self._lod]
        self._trans_covar_lower = trans_covar[:, self._lod:]
        
        #Calling the build function for the Layer superclass
        super().build(input_shape)

    def _predict(self, post_mean, post_covar):

    def _update(self, prior_mean, prior_covar, obs_mean, obs_covar):

    def get_initial_state(self, inputs, bath_size, dtype):

    @property
    #Required for use with tf.keras.layers.RNN
    def state_size(self):
        return 5*self._lod





