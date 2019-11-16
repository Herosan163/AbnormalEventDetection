import numpy as np
import tensorflow as tf
from model import convolutional_auto_encoder
import os
from PIL import Image
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from joblib import dump, load


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    n_clusters = 10
    # data, grad_former, grad_later = create_dataset(threshold)
    np.save('data.npy', data)
    np.save('grad_former.npy', grad_former)
    np.save('grad_later.npy', grad_later)
    data = np.load('data.npy')
    grad_former = np.load('grad_former.npy')
    grad_later = np.load('grad_later.npy')
    data = np.expand_dims(data, axis=-1)
    grad_former = np.expand_dims(grad_former, axis=-1)
    grad_later = np.expand_dims(grad_later, axis=-1)
    train_ds = zip(data, grad_former, grad_later)
    model_appearance = convolutional_auto_encoder()
    model_appearance.compile(loss=tf.keras.losses.MSE,
                             optimizer=tf.keras.optimizers.Adam())
    model_appearance.train_on_batch(data[:1], data[:1])
    model_motion1 = convolutional_auto_encoder()
    model_motion1.compile(loss=tf.keras.losses.MSE,
                          optimizer=tf.keras.optimizers.Adam())
    model_motion1.train_on_batch(grad_former[:1], grad_former[:1])
    model_motion2 = convolutional_auto_encoder()
    model_motion2.compile(loss=tf.keras.losses.MSE,
                          optimizer=tf.keras.optimizers.Adam())
    model_motion2.train_on_batch(grad_later[:1], grad_later[:1])
    model_appearance.load_weights(os.path.join(args.result_directory, 'model_appearance_best_loss.h5'))
    model_motion1.load_weights(os.path.join(args.result_directory, 'model_motion1_best_loss.h5'))
    model_motion2.load_weights(os.path.join(args.result_directory, 'model_motion2_best_loss.h5'))
    features = []
    for img, gradient1, gradient2 in train_ds:
        encode_appearance = model_appearance.encode(img[None]).numpy()
        encode_motion1 = model_motion1.encode(gradient1[None]).numpy()
        encode_motion2 = model_motion2.encode(gradient2[None]).numpy()
        feature = np.r_[encode_appearance.flatten(), encode_motion1.flatten(), encode_motion2.flatten()]
        features.append(feature)
    np.save('features.npy', np.array(features))
    features = np.load('features.npy')
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    pred_km = km.fit_predict(features)
    np.save('pred_km.npy', pred_km)
    pred_km = np.load('pred_km.npy')
    for i in range(n_clusters):
        if i < 5:
            continue
        clf = SVC(C=1.0, kernel='linear')
        y = np.zeros((features.shape[0],))
        indices = np.where(pred_km==i)[0]
        y[indices] = 1
        clf.fit(features, y)
        dump(clf, os.path.join(args.result_directory, 'svm_%d.pickle' % (i)))
