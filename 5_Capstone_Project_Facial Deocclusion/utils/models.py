import tensorflow as tf
from tensorflow import keras

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Add, LeakyReLU
from keras import Model

from utils.define_class import *


def Generator(shape=(256,256,3), train_attention=True):
    x_input = tf.keras.layers.Input(shape=shape, name='Input_layer')

    encoder0 = Residual_Block(num_filter=32, kernel_size=3, name="Encoder_0")(x_input)
    encoder0 = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPooling2D_0")(encoder0)

    encoder1 = Residual_Block(num_filter=64, kernel_size=3, name="Encoder_1")(encoder0)
    encoder1 = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPooling2D_1")(encoder1)
    
    encoder2 = Residual_Block(num_filter=128, kernel_size=3, name="Encoder_2")(encoder1)
    maxpooling_2 = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPooling2D_2")(encoder2)

    encoder3 = Residual_Block(num_filter=256, kernel_size=3, name="Encoder_3")(maxpooling_2)
    encoder3 = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPooling2D_3")(encoder3)

    encoder4 = Residual_Block(num_filter=512, kernel_size=3, name="Encoder_4")(encoder3)
    encoder4 = tf.keras.layers.MaxPooling2D((2, 2), name="MaxPooling2D_4")(encoder4)

    bottleneck = tf.keras.layers.Conv2D(1024, 3, padding="same", name="bottleneck")(encoder4)

    decoder0 = Decoder_Block(num_filter=512, kernel_size=3, name="Decoder_0")(bottleneck)
    decoder1 = Decoder_Block(num_filter=256, kernel_size=3, name="Decoder_1")(decoder0)
    decoder2 = Decoder_Block(num_filter=128, kernel_size=3, name="Decoder_2")(decoder1)
    if train_attention:
        f_enc = spatial_attention(trainable=train_attention, name="F_enc")(encoder2)
        f_dec = spatial_attention(trainable=train_attention, name="F_dec")(decoder2)
        f_fused = compute_fused(f_enc, f_dec, trainable=train_attention, name_layer="Attention_map")
        input_decoder3 = tf.keras.layers.Add(name="Add_f_fused")([decoder2, f_fused])
        decoder3 = Decoder_Block(num_filter=64, kernel_size=3, name="Decoder_3")(input_decoder3)
        
    else:
        decoder3 = Decoder_Block(num_filter=64, kernel_size=3, name="Decoder_3")(decoder2)
    decoder4 = Decoder_Block(num_filter=32, kernel_size=3, name="Decoder_4")(decoder3)

    imageout = Ouput_layer(name="Ouput_layer")(decoder4)

    mymodel = tf.keras.Model(inputs=x_input, outputs=imageout, name="Generator")
    return mymodel


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (e, 256, 256, channels*2)batch_siz

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
