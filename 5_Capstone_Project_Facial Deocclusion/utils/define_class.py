
import tensorflow as tf
from tensorflow import keras

class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=7, trainable=True ,**kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv2D(filters=1, 
                                             kernel_size=self.kernel_size,
                                             strides=1, 
                                             padding='same', 
                                             activation='sigmoid',
                                             kernel_initializer='he_normal', 
                                             use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)
        multiplied = tf.keras.layers.Multiply()([inputs, feature])
        # shape_out = multiplied.shape
        # return tf.keras.layers.multiply()([inputs, feature])
        return multiplied
    

class Attention_map(tf.keras.layers.Layer):
    """ 
    Attention module
    As described in: https://arxiv.org/pdf/2112.01098.pdf
    """ 
    def __init__(self,num_filters, trainable=True, **kwargs):
        self.num_filters = num_filters
        self.initializer = tf.random_normal_initializer(0., 0.02)

        self.conv4a = tf.keras.layers.Conv2D(4*self.num_filters, 
                                            kernel_size=3,
                                            padding="same")

        self.conv8 = tf.keras.layers.Conv2D(8*self.num_filters, 
                                            kernel_size=3,
                                            padding="same")

        self.conv4b = tf.keras.layers.Conv2D(4*self.num_filters, 
                                            kernel_size=3,
                                            padding="same")

        self.conv2 = tf.keras.layers.Conv2D(2*self.num_filters, 
                                            kernel_size=3,
                                            padding="same") 

        super(Attention_map,self).__init__(**kwargs)

    def call(self, inputs, training, **kwargs):
        x = self.conv4a(inputs)
        x = self.conv8(x)
        x = self.conv4b(x)
        x = self.conv2(x)
        return x

    
def compute_fused(fenc, fdec, name_layer, trainable=True):
    """
    Compute fused module \n
    f_fused = fenc*Attention_map[0] + fdec*Attention_map[1]
    """
    fconcat =  tf.concat([fenc, fdec], 0)
    output_attentionmap = Attention_map(num_filters=64,trainable=trainable, name=name_layer)(fconcat)
    f_fused = fenc*output_attentionmap[0] + fdec*output_attentionmap[1]
    return f_fused

class Residual_Block(keras.layers.Layer):
    """
    Residual Block from ResNet:
    Input: (2H,2W,C) => Ouput: (H,W,C')
    """
    def __init__(self, num_filter, kernel_size=3, trainable=True,  **kwargs):
        super(Residual_Block, self).__init__(**kwargs)
        self.filters= num_filter
        self.kernel_size = kernel_size
        self.trainable = trainable

    def build(self, input_shape):
        self.x_skip = tf.keras.layers.Conv2D(self.filters,
                                             kernel_size=1,
                                             padding="same")

        self.conv2a = tf.keras.layers.Conv2D(self.filters, 
                                             kernel_size=self.kernel_size, 
                                             padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters=self.filters, 
                                             kernel_size=self.kernel_size,
                                             padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        super(Residual_Block, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        x_skip = self.conv2a(inputs)

        x = self.conv2a(inputs)
        x = self.bn2a(x,self.trainable)
        out1 = tf.nn.relu(x)

        out2 = self.conv2b(out1)
        out2 = self.bn2b(out2,self.trainable)
        out2 = tf.nn.relu(out2)


        add = tf.keras.layers.Add()([x_skip, out2])
        out = tf.nn.relu(add)
        return out


class Decoder_Block(tf.keras.layers.Layer):
    """
    Decoder Block: Upsampling and Residual Block 
    Input: (H,W, 2C) => Ouput: (2H,2W, C)
    """
    def __init__(self,num_filter, kernel_size=3, trainable=True, **kwargs):
        super(Decoder_Block, self).__init__(**kwargs)
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.residual = Residual_Block(num_filter=self.num_filter, 
                                       kernel_size=self.kernel_size)
        self.upsampling = tf.keras.layers.Conv2DTranspose(filters=self.num_filter,
                                                          kernel_size=2,
                                                          strides=2)
    def call(self, inputs, **kwargs):
        x = self.upsampling(inputs)
        x = self.residual(x)
        return x

class Ouput_layer(tf.keras.layers.Layer):
    """
    Output layer of model with shape (H,W,3)
    """
    def __init__(self, trainable=True, **kwargs):
        super(Ouput_layer, self).__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(filters=3,
                                             kernel_size=3,
                                             padding="same")
        self.bn2d = tf.keras.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        output = self.conv2d(inputs)
        output = self.bn2d(output)
        output = tf.nn.relu(output)
        return output

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result