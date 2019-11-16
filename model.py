import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D

class convolutional_auto_encoder(Model):
    def __init__(self):
        super(convolutional_auto_encoder, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', padding='SAME')
        self.pool2 = MaxPooling2D(2, 2)
        self.conv3 = Conv2D(32, 3, activation='relu', padding='SAME')
        self.pool4 = MaxPooling2D(2, 2)
        self.conv5 = Conv2D(16, 3, activation='relu', padding='SAME')
        self.pool6 = MaxPooling2D(2, 2)
        self.conv7 = Conv2D(16, 3, activation='relu', padding='SAME')
        self.upsample8 = UpSampling2D(2, interpolation='nearest')
        self.conv9 = Conv2D(32, 3, activation='relu', padding='SAME')
        self.upsample10 = UpSampling2D(2, interpolation='nearest')
        self.conv11 = Conv2D(32, 3, activation='relu', padding='SAME')
        self.upsample12 = UpSampling2D(2, interpolation='nearest')
        self.conv13 = Conv2D(1, 3, padding='SAME')

    @tf.function
    def calc_loss(self, x):
        x = tf.cast(x, tf.float32)
        x = x / 255.0
        decode = self(x)        
        loss = tf.reduce_mean(tf.square(x - decode))
        return loss

    @tf.function
    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = x / 255.0
        encoded_feature = self.encode(x)
        decoded_featrue = self.decode(encoded_feature)
        return decoded_featrue

    @tf.function
    def encode(self, x):
        x = tf.cast(x, tf.float32)
        x = x / 255.0
        x = self.conv1(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool6(x)
        return x

    def decode(self, x):
        x = self.conv7(x)
        x = self.upsample8(x)
        x = self.conv9(x)
        x = self.upsample10(x)
        x = self.conv11(x)
        x = self.upsample12(x)
        x = self.conv13(x)
        return x
