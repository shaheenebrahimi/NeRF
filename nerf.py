from load_data import *
# import tensorflow.keras as keras
import keras
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# NeRF model
class NeRF(keras.Model):
    def __init__(self, embedded_x_dim=10, embedded_d_dim=4):
        '''Initialize network'''
        super(NeRF, self).__init__()
        self.embedded_x_dim = embedded_x_dim
        self.embedded_d_dim = embedded_d_dim
        self.gamma_x = embedded_x_dim * 3 # 3 dimensional, taylor expansion (sin, cos)
        self.gamma_d = embedded_d_dim * 3
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
    def positional_encoding(pos, L):
        '''Encode ray to sinusoidal harmonics'''
        result = []
        for i in range(L//2):
            result.append(tf.sin(2**i * pos))
            result.append(tf.cos(2**i * pos))
        return tf.concat(result, axis=-1)
    
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

def accumulated_transmittance(alphas):
    '''Compute accumulated transmittance'''
    transmittance = tf.math.cumprod(alphas, axis=1, exclusive=True)
    ones = tf.ones([tf.shape(transmittance)[0], 1], dtype = alphas.dtype)
    return tf.concat([ones, transmittance[:, :-1]], axis=-1)


def render(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    '''Render image based on camera ray origins and ray directions'''
    # Sample points along rays, linear space between hn and hf
    t = tf.cast(tf.tile(tf.linspace(hn, hf, nb_bins)[None, :], [ray_origins.shape[0], 1]), dtype=tf.dtypes.float32) # expand to match the number of rays
    
    # Get lower and upper bounds for each bin
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = tf.concat([t[:, :1], mid], axis=-1)
    upper = tf.concat([mid, t[:, -1:]], axis=-1)
    u = tf.random.uniform(tf.shape(t))   
    t = lower + (upper - lower) * u # parametric representation

    # Delta for transmittance calculation
    const = tf.repeat(tf.constant([1e10], dtype=tf.dtypes.float32)[None, :], ray_origins.shape[0], axis=0)
    delta = tf.concat([t[:, 1:] - t[:, :-1], const], axis=-1)

    # Compute 3D points along each ray
    x = ray_origins[:, None, :] + t[:, :, None] * ray_directions[:, None, :] # sampled point
    ray_directions = tf.repeat(ray_directions[:, None, :], nb_bins, axis=1) # resize for batching

    # Model Prediction
    colors, sigma = nerf_model(tf.reshape(x, [-1, 3]), tf.reshape(ray_directions, [-1, 3]))
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

def train(nerf_model, optimizer, data, epochs=int(1e5), hn=0, hf=1, nb_bins=192, H=400, W=400):
    # Clear log file
    with open('logs/log.txt', 'w') as log_file:
        log_file.write('')

    training_loss = []
    for i in range(epochs):
        for batch in tqdm(data, desc='Epoch ' + str(i), ncols=100):
            ray_origins = batch[:, :3]
            ray_directions = batch[:, 3:6]
            y = batch[:, 6:] # ground truth

            # Record forward pass with tape
            with tf.GradientTape() as tape:
                logits = render(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
                loss = tf.math.reduce_sum((y - logits) ** 2)
            
            # Compute and apply gradients
            grads = tape.gradient(loss, nerf_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Output loss to logfile
            with open('log.txt', 'a') as log_file:
                log_file.write(str(loss.numpy()) + '\n')
        
        training_loss.append(loss)
        model.save('nuthin.keras')

    log_file.close()
    return training_loss

def test(nerf_model, hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        nb_bins (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.
        
    Returns:
        None: None
    """
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = [] # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        # Get chunk of rays
        o = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size]
        d = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size]   
        generated_px_values = render(nerf_model, ray_origins=o, ray_directions=d, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(generated_px_values)
    img = np.array(data).reshape(H, W, 3)

    plt.figure()
    # plt.imshow(img)
    # plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.imsave(f'novel_views/img_{img_index}.png', img)
    plt.close()

if __name__ == '__main__':
    # Deine Constants
    EPOCHS = 5
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-5

    # Load data
    train_dataset = np.load('parsed_data/training_data.pkl', allow_pickle=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(BATCH_SIZE).batch(BATCH_SIZE)

    test_dataset = np.load('parsed_data/testing_data.pkl', allow_pickle=True)
    # test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
    
    # Create and train model
    model = NeRF()
    # train(model, tf.optimizers.legacy.Adam(LEARNING_RATE), train_dataset, epochs=EPOCHS)
    test(model,0,1,test_dataset)


