import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D

class convolutional_auto_encoder(Model):
    def __init__(self):
        super(convolutional_auto_encoder, self).__init__()
        self.encode_net = tf.keras.Sequential(
            [
                Conv2D(32, 3, activation='relu', padding='SAME'),
                MaxPooling2D(2, 2),
                Conv2D(32, 3, activation='relu', padding='SAME'),
                MaxPooling2D(2, 2),
                Conv2D(16, 3, activation='relu', padding='SAME'),
                MaxPooling2D(2, 2)
            ]
        )
        self.decode_net = tf.keras.Sequential(
            [
                Conv2D(16, 3, activation='relu', padding='SAME'),
                UpSampling2D(2, interpolation='nearest'),
                Conv2D(32, 3, activation='relu', padding='SAME'),
                UpSampling2D(2, interpolation='nearest'),
                Conv2D(32, 3, activation='relu', padding='SAME'),
                UpSampling2D(2, interpolation='nearest'),
                Conv2D(1, 3, padding='SAME')
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
