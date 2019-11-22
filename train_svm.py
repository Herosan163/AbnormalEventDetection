import numpy as np
import tensorflow as tf
from model import convolutional_auto_encoder
import os
from PIL import Image
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from joblib import dump, load

tf.compat.v1.enable_eager_execution()

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    n_clusters = 10
    data = np.load('data.npy')
    data_former = np.load('data_former.npy')
    data_later = np.load('data_later.npy')
    data = np.expand_dims(data, axis=-1)
    data_former = np.expand_dims(data_former, axis=-1)
    data_later = np.expand_dims(data_later, axis=-1)
    train_ds = zip(data, data_former, data_later)
    model_appearance = convolutional_auto_encoder()
    model_appearance.compile(loss=tf.keras.losses.MSE,
                             optimizer=tf.keras.optimizers.Adam())
    model_appearance.train_on_batch(data[:1].astype(np.float32), data[:1].astype(np.float32))
    model_motion1 = convolutional_auto_encoder()
    model_motion1.compile(loss=tf.keras.losses.MSE,
                          optimizer=tf.keras.optimizers.Adam())
    model_motion1.train_on_batch(data_former[:1].astype(np.float32), data_former[:1].astype(np.float32))
    model_motion2 = convolutional_auto_encoder()
    model_motion2.compile(loss=tf.keras.losses.MSE,
                          optimizer=tf.keras.optimizers.Adam())
    model_motion2.train_on_batch(data_later[:1].astype(np.float32), data_later[:1].astype(np.float32))
    model_appearance.load_weights(os.path.join(args.result_directory, 'model_appearance_best_loss.hdf5'))
    model_motion1.load_weights(os.path.join(args.result_directory, 'model_motion1_best_loss.hdf5'))
    model_motion2.load_weights(os.path.join(args.result_directory, 'model_motion2_best_loss.hdf5'))
    features = []
    for img, former, later in train_ds:
        img = tf.cast(tf.identity(img), tf.float32)
        former = tf.cast(tf.identity(former), tf.float32)
        later = tf.cast(tf.identity(later), tf.float32)
        encode_appearance = model_appearance.encode(img[None] / 255.0).numpy()
        encode_motion1 = model_motion1.encode((img - former)[None] / 255.0).numpy()
        encode_motion2 = model_motion2.encode((later - img)[None] / 255.0).numpy()
        feature = np.r_[encode_appearance.flatten(), encode_motion1.flatten(), encode_motion2.flatten()]
        features.append(feature)
    features = np.array(features)
    np.save('features.npy', features)
    # features = np.load('features.npy')
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    pred_km = km.fit_predict(features)
    np.save('pred_km.npy', pred_km)
    # pred_km = np.load('pred_km.npy')
    clf = LinearSVC(C=1.0, multi_class='ovr', max_iter=100000, loss='hinge')
    clf.fit(features, pred_km)
    dump(clf, os.path.join(args.result_directory, 'svm.pickle'))
