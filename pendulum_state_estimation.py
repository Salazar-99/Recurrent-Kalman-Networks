import numpy as np
import tensorflow as tf 
from pendulum_data import Pendulum
from models import RKN


def generate_pendulum_filter_dataset(pendulum, num_seqs, seq_length, seed):
    obs, targets, _, _ = pendulum.sample_data_set(num_seqs, seq_length, full_targets=False, seed=seed)
    obs, _ = data.add_observation_noise(obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
    obs = np.expand_dims(obs, -1)
    return obs, targets

# Implement Encoder and Decoder hidden layers
class PendulumStateEstemRKN(RKN):

    def build_encoder_hidden(self):
        return [
            # 1: Conv Layer
            tf.keras.layers.Conv2D(12, kernel_size=5, padding="same"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(2, strides=2),
            # 2: Conv Layer
            tf.keras.layers.Conv2D(12, kernel_size=3, padding="same", strides=2),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(2, strides=2),
            tf.keras.layers.Flatten(),
            # 3: Dense Layer
            tf.keras.layers.Dense(30, activation=tf.keras.activations.relu)]

    def build_decoder_hidden(self):
        return [tf.keras.layers.Dense(units=10, activation=tf.keras.activations.relu)]

    def build_var_decoder_hidden(self):
        return [tf.keras.layers.Dense(units=10, activation=tf.keras.activations.relu)]


# Generate Data
pend_params = Pendulum.pendulum_default_params()
pend_params[Pendulum.FRICTION_KEY] = 0.1
data = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                transition_noise_std=0.1,
                observation_noise_std=1e-5,
                seed=0,
                pendulum_params=pend_params)

train_obs, train_targets = generate_pendulum_filter_dataset(data, 200, 75, np.random.randint(1e8))
test_obs, test_targets = generate_pendulum_filter_dataset(data, 100, 75, np.random.randint(1e8))

# Build Model
rkn = PendulumStateEstemRKN(observation_shape=train_obs.shape[-3:], latent_observation_dim=15, output_dim=2, num_basis=15,
                            bandwidth=3)
rkn.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=5.0), loss=RKN.gaussian_neg_log_likelihood, metrics=[RKN.rmse])

# Train Model
rkn.fit(train_obs, train_targets, batch_size=50, epochs=500, validation_data=(test_obs, test_targets), verbose=2)








