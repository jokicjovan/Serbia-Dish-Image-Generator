import numpy as np
import tensorflow as tf
from keras import Model
from keras.initializers import he_normal
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Flatten, Dense, InputLayer, Reshape
from keras.src.losses import binary_crossentropy


class Encoder(Model):

    def __init__(self, latent_dim, concat_input_and_condition=True):

        super(Encoder, self).__init__()
        self.use_cond_input = concat_input_and_condition
        self.enc_block_1 = Conv2D(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_2 = Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_3 = Conv2D(
            filters=128,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_4 = Conv2D(
            filters=256,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_5 = Conv2D(
            filters=512,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.bn5 = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()

        self.flatten = Flatten()
        self.dense = Dense(2* latent_dim)

    def __call__(self, input_img, input_label, conditional_input, is_train):
        x = conditional_input if self.use_cond_input else input_img

        x = self.enc_block_1(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.enc_block_2(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.enc_block_3(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.enc_block_4(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.enc_block_5(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        if not self.use_cond_input:
            cond = tf.reshape(input_label, [tf.shape(input_img)[0], 4, 4, -1])
            x = tf.concat([x, cond], axis=3)

        x = self.dense(self.flatten(x))

        return x

class Decoder(Model):

    def __init__(self, batch_size=32):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.dense = Dense(4 * 4 * 512)
        self.reshape = Reshape(target_shape=(4, 4, 512))

        self.dec_block_1 = Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_2 = Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_3 = Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_4 = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_5 = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_out = Conv2DTranspose(
            filters=3,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            kernel_initializer=he_normal())

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.bn5 = BatchNormalization()

    def __call__(self, z_cond, is_train):
        x = self.dense(z_cond)
        x = tf.nn.leaky_relu(x)
        x = self.reshape(x)

        x = self.dec_block_1(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.dec_block_2(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.dec_block_3(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.dec_block_4(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.dec_block_5(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        return self.dec_block_out(x)


class ConvCVAE(Model):

    def __init__(self,
                 encoder,
                 decoder,
                 label_dim,
                 latent_dim,
                 batch_size=32,
                 beta=1,
                 image_dim= [128,128,3]):
        super(ConvCVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta
        self.image_dim = image_dim

    def __call__(self, inputs, is_train):
        input_img, input_label, conditional_input = self.conditional_input(inputs)

        z_mean, z_log_var = tf.split(self.encoder(input_img, input_label, conditional_input, is_train), num_or_size_splits=2, axis=1)
        z_cond = self.reparametrization(z_mean, z_log_var, input_label)
        logits = self.decoder(z_cond, is_train)

        recon_img = tf.nn.sigmoid(logits)

        latent_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                                            axis=-1)
        reconstr_loss = np.prod((128, 128)) * binary_crossentropy(Flatten()(input_img), Flatten()(recon_img))  # over weighted MSE
        loss = reconstr_loss + self.beta * latent_loss  # weighted ELBO loss
        loss = tf.reduce_mean(loss)

        return {
            'recon_img': recon_img,
            'latent_loss': latent_loss,
            'reconstr_loss': reconstr_loss,
            'loss': loss,
            'z_mean': z_mean,
            'z_log_var': z_log_var
        }

    def conditional_input(self, inputs):
        input_img = inputs[0]
        input_label = inputs[1]
        labels = tf.reshape(input_label, [-1, 1, 1, self.label_dim])
        ones = tf.ones([tf.shape(input_img)[0]] + self.image_dim[0:-1] + [self.label_dim])
        labels = ones * labels
        conditional_input = tf.concat([input_img, labels], axis=3)

        return input_img, input_label, conditional_input

    def reparametrization(self, z_mean, z_log_var, input_label):
        batch_size = tf.shape(input_label)[0]
        eps = tf.random.normal(shape=(batch_size, self.latent_dim), mean=0.0, stddev=1.0)
        z = z_mean + tf.math.exp(z_log_var * .5) * eps
        z_cond = tf.concat([z, input_label], axis=1)  # (batch_size, label_dim + latent_dim)

        return z_cond

    def generate(self, text_embeddings, num_samples=None):
        if len(text_embeddings.shape) == 1:
            num_samples = num_samples or 1
            text_embeddings = tf.tile(tf.expand_dims(text_embeddings, 0), [num_samples, 1])

        batch_size = tf.shape(text_embeddings)[0]

        z = tf.random.normal(shape=(batch_size, self.latent_dim), mean=0.0, stddev=1.0)

        z_cond = tf.concat([z, text_embeddings], axis=1)

        logits = self.decoder(z_cond, is_train=False)
        generated_images = tf.nn.sigmoid(logits)

        return generated_images

    def reconstruct(self, images, labels):
        output = self((images, labels), is_train=False)
        return output['recon_img']


