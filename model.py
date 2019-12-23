import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D

class convolutional_auto_encoder(Model):
    def __init__(self, name):
        super(convolutional_auto_encoder, self).__init__()
        self.name_prefix = name
        self.encode_net = tf.keras.Sequential(
            [
                Conv2D(32, 3, activation='relu', padding='SAME', name=name+'_conv1'),
                MaxPooling2D(2, 2, padding='SAME', name=name+'_'+'pool1'),
                Conv2D(32, 3, activation='relu', padding='SAME', name=name+'_conv2'),
                MaxPooling2D(2, 2, padding='SAME', name=name+'_'+'pool2'),
                Conv2D(16, 3, activation='relu', padding='SAME', name=name+'_conv3'),
                MaxPooling2D(2, 2, padding='SAME', name=name+'_pool3')
            ]
        )
        self.decode_net = tf.keras.Sequential(
            [
                UpSampling2D(2, interpolation='nearest', name=name+'_upsampling1'),
                Conv2D(16, 3, activation='relu', padding='SAME', name=name+'_conv4'),
                UpSampling2D(2, interpolation='nearest', name=name+'_upsampling2'),
                Conv2D(32, 3, activation='relu', padding='SAME', name=name+'_conv5'),
                UpSampling2D(2, interpolation='nearest', name=name+'_upsampling3'),
                Conv2D(32, 3, activation='relu', padding='SAME', name=name+'_conv6'),
                Conv2D(1, 3, padding='SAME', name=name+'_conv7')
            ]
        )

    def call(self, x):
        encode = self.encode_net(x)
        decode = self.decode_net(encode)
        return encode, decode
    
    @tf.function
    def extract_feature(self, x):
        return self.encode_net(x)

    def encode(self, x):
        return self.encode_net(x)

    def decode(self, x):
        return self.decode_net(x)
