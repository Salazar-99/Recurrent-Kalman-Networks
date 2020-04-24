import numpy as np 
import tensorflow as tf
import math as m

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
        #TODO: Implement pack_input
        rkn_in = pack_input(w_mean, w_covar)
        z = self._layer_rkn(rkn_in)
        #TODO: Implement unpack_state(z)
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

    def build_var_Decoder_hidden(self):
        rasie NotImplementedError




