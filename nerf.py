from load_data import *
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

# NeRF model
class NeRF(keras.Model):
    def __init__(self, embedded_x_dim=10, embedded_d_dim=4):
        '''Initialize network'''
        super(NeRF, self).__init__()
        self.embedded_x_dim = embedded_x_dim
        self.embedded_d_dim = embedded_d_dim
        self.gamma_x = embedded_x_dim * 3 * 2 # 3 dimensional, taylor expansion (sin, cos)
        self.gamma_d = embedded_d_dim * 3 * 2
        self.block1 = keras.Sequential(
            [ keras.layers.Input(shape=(self.gamma_x,)) ] + [ keras.layers.Dense(256, activation='relu') for i in range(4) ]
        )
        self.block2 = keras.Sequential(
            [ keras.layers.Input(shape=(self.gamma_x + 256,)) ] + [ keras.layers.Dense(256, activation='relu') for i in range(4) ]
        )
        self.block3 = keras.Sequential([
            keras.layers.Input(shape=(self.gamma_d + 256,)),
            keras.layers.Dense(128, activation='relu')
        ])
        self.output1 = keras.Sequential([
            keras.layers.Input(shape=(256,)),
            keras.layers.Dense(1, activation='relu')
        ])
        self.output2 = keras.Sequential([
            keras.layers.Input(shape=(128,)),
            keras.layers.Dense(3, activation='sigmoid')
        ])

    @staticmethod
    def positional_encoding(p, L):
        '''p: input, L: output dimensionality'''
        embedding = np.zeros(shape=(2*L,))
        for i in range(L):
            embedding[2*i] = np.sin(2**i * np.pi * p)
            embedding[2*i + 1] = np.cos(2**i * np.pi * p)
        return tf.convert_to_tensor(embedding)
    
    def call(self, x, d):
        '''Forward propagation call'''
        embedded_x = self.positional_encoding(x, self.embedded_x_dim)
        embedded_d = self.positional_encoding(d, self.embedded_d_dim)
        z = self.block1(embedded_x)
        z = self.block2(keras.layers.Concatenate()([z, embedded_x]))
        sigma = self.output1(z)
        z = self.block3(keras.layers.Concatenate()([z, embedded_d]))
        color = self.output2(z)
        return color, sigma

    def render(self, ray_origins, ray_directions):
        '''Render image based on camera ray origins and ray directions'''

def accumulated_transmittance(alphas):
    transmittance = tf.math.cumprod(alphas, axis=1, exclusive=True)
    ones = tf.ones([tf.shape(transmittance)[0], 1], dtype = alphas.dtype)
    return tf.concat([ones, transmittance[:, :-1]], axis=-1)


def render(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    
    # Sample points along rays, linear space between hn and hf
    t = tf.tile(tf.linspace(hn, hf, nb_bins)[None, :], [ray_origins.shape[0], 1]) # expand to match the number of rays
    # t = tf.repeat(tf.linspace(hn, hf, nb_bins)[None, :], ray_origins.shape[0], axis=0) # effectively equivalent to above

    # Random perturbation
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = tf.concat([t[:, :1], mid], axis=-1)
    upper = tf.concat([mid, t[:, -1:]], axis=-1)
    u = tf.random.uniform(tf.shape(t))
    t = lower + (upper - lower) * u 

    # Delta for transmittance calculation
    const = tf.repeat(tf.constant([1e10])[None, :], ray_origins.shape[0], axis=0)
    delta = tf.concat([t[:, 1:] - t[:, :-1], const], axis=-1)

    # Compute 3D points along each ray
    # x = tf.expand_dims(ray_origins, 1) + tf.expand_dims(t, 2) * tf.expand_dims(ray_directions, 1) # in case below doesn't work
    x = ray_origins[:, None, :] + t[:, :, None] * ray_directions[:, None, :]
    ray_dirs = tf.repeat(ray_directions[:, None, :], nb_bins, axis=1)

    # Model Prediction
    colors, sigma = nerf_model([tf.reshape(x, [-1, 3]), tf.reshape(ray_dirs, [-1, 3])])
    colors = tf.reshape(colors, tf.shape(x))
    sigma = tf.reshape(sigma, tf.shape(x)[:-1])

    # Compute alpha and weights
    alpha = 1 - tf.exp(-sigma * delta)
    transmittance = accumulated_transmittance(1 - alpha)
    weights = tf.expand_dims(transmittance, 2) * tf.expand_dims(alpha, 2)

    # Compute final color as weighted sum of colors along each ray
    c = tf.reduce_sum(weights * colors, axis=1)
    weight_sum = tf.reduce_sum(tf.reduce_sum(weights, axis=-1), axis=-1) # regularization of white background

    return c + (1 - tf.expand_dims(weight_sum, -1))


if __name__ == '__main__':
    model = NeRF()