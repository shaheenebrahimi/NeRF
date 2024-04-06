from load_data import *
import keras
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

if __name__ == '__main__':
    model = NeRF()