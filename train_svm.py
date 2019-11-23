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
    data = np.expand_dims(data, axis=-1).astype(np.float32)
    data_former = np.expand_dims(data_former, axis=-1).astype(np.float32)
    data_later = np.expand_dims(data_later, axis=-1).astype(np.float32)
    data /= 255.
    data_former /= 255.
    data_later /= 255.
    train_ds = zip(data, data_former, data_later)
    input_shape = (None, 64, 64, 1)
    model_appearance = convolutional_auto_encoder()
    model_appearance.build(input_shape)
    model_motion1 = convolutional_auto_encoder()
    model_motion1.build(input_shape)
    model_motion2 = convolutional_auto_encoder()
    model_motion2.build(input_shape)
    model_appearance.load_weights(os.path.join(args.result_directory, 'model_appearance.h5'))
    model_motion1.load_weights(os.path.join(args.result_directory, 'model_motion1.h5'))
    model_motion2.load_weights(os.path.join(args.result_directory, 'model_motion2.h5'))
    # img = Image.fromarray((data[0,:,:,0]*255).astype(np.uint8))
    # img.save('input.png')    
    # encode = model_appearance.extract_feature(data[0][None])
    # decode = model_appearance.decode(encode).numpy()
    # img = Image.fromarray((decode[0,:,:,0]*255).astype(np.uint8))
    # img.save('decode.png')
    features = []
    for img, former, later in train_ds:
        encode_appearance = model_appearance.extract_feature(img[None]).numpy()
        encode_motion1 = model_motion1.extract_feature((img - former)[None]).numpy()
        encode_motion2 = model_motion2.extract_feature((later - img)[None]).numpy()
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
