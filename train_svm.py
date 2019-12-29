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
from cyvlfeat.kmeans import kmeans, kmeans_quantize
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

tf.compat.v1.enable_eager_execution()

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    return parser.parse_args()

def create_gradient(data, data_former, data_later):

    grad1 = []
    grad2 = []
    data = data.astype(np.float32)
    data_former = data_former.astype(np.float32)
    data_later = data_later.astype(np.float32)
    for i in range(data.shape[0]):
        grad1.append(data[i] - data_former[i])
        grad2.append(data_later[i] - data[i])
    return data, np.array(grad1), np.array(grad2)


if __name__ == '__main__':

    args = get_args()
    n_clusters = 10
    data = np.load('data.npy')
    data_former = np.load('data_former.npy')
    data_later = np.load('data_later.npy')
    data, former_grad, later_grad = create_gradient(data, data_former, data_later)
    data = np.expand_dims(data, axis=-1).astype(np.float32)
    former_grad = np.expand_dims(former_grad, axis=-1).astype(np.float32)
    later_grad = np.expand_dims(later_grad, axis=-1).astype(np.float32)
    data /= 255.
    former_grad /= 255.
    later_grad /= 255.
    train_ds = zip(data, former_grad, later_grad)
    input_shape = (None, 64, 64, 1)
    model_appearance = convolutional_auto_encoder(name='appearance')
    model_motion1 = convolutional_auto_encoder(name='motion1')
    model_motion2 = convolutional_auto_encoder(name='motion2')
    model_appearance.build(input_shape)
    model_motion1.build(input_shape)
    model_motion2.build(input_shape)
    model_appearance.load_weights(os.path.join(args.result_directory, 'model_appearance_epoch199.h5'))
    model_motion1.load_weights(os.path.join(args.result_directory, 'model_motion1_epoch199.h5'))
    model_motion2.load_weights(os.path.join(args.result_directory, 'model_motion2_epoch199.h5'))
    # im = np.array(Image.open('input_gray.png'))
    # former = np.array(Image.open('input_former.png'))
    # later = np.array(Image.open('input_later.png'))
    # img = tf.cast(tf.identity(np.expand_dims(im, axis=-1)), tf.float32)
    # former = tf.cast(tf.identity(np.expand_dims(former, axis=-1)), tf.float32)
    # later = tf.cast(tf.identity(np.expand_dims(later, axis=-1)), tf.float32)
    # encode_appearance = model_appearance.extract_feature(img[None] / 255.0)
    # encode_motion1 = model_motion1.extract_feature((former[None]-100) / 255.0)
    # encode_motion2 = model_motion2.extract_feature((later[None]-100) / 255.0)
    # decode_gray = model_appearance.decode(encode_appearance).numpy()
    # img = Image.fromarray((decode_gray[0,:,:,0]*255).astype(np.uint8))
    # img.save('decode_gray.png')
    # decode_motion1 = model_motion1.decode(encode_motion1).numpy()
    # img = Image.fromarray((decode_motion1[0,:,:,0]*255+100).astype(np.uint8))
    # img.save('decode_motion1.png')
    # decode_motion2 = model_motion1.decode(encode_motion2).numpy()
    # img = Image.fromarray((decode_motion2[0,:,:,0]*255+100).astype(np.uint8))
    # img.save('decode_motion2.png')

    features = []
    i=0
    for img, grad1, grad2 in train_ds:
        i+=1
        if not i%500==0:
            continue
        encode_appearance = model_appearance.extract_feature(img[None]).numpy()
        encode_motion1 = model_motion1.extract_feature(grad1[None]).numpy()
        encode_motion2 = model_motion2.extract_feature(grad2[None]).numpy()
        feature = np.r_[encode_appearance.flatten(), encode_motion1.flatten(), encode_motion2.flatten()]
        features.append(feature)
        decode_gray = model_appearance.decode(encode_appearance).numpy()
        img = Image.fromarray((decode_gray[0,:,:,0]*255).astype(np.uint8))
        img.save('check2/decode_gray_%d.png'%i)
        grad1 = Image.fromarray(((grad1[:,:,0]*255+255)/2).astype(np.uint8))
        grad1.save('check2/input_motion1_%d.png'%i)
        decode_motion1 = model_motion1.decode(encode_motion1).numpy()
        img = Image.fromarray(((decode_motion1[0,:,:,0]*255+255)/2).astype(np.uint8))
        img.save('check2/decode_motion1_%d.png'%i)
        grad2 = Image.fromarray(((grad2[:,:,0]*255+255)/2).astype(np.uint8))
        grad2.save('check2/input_motion2_%d.png'%i)
        decode_motion2 = model_motion1.decode(encode_motion2).numpy()
        img = Image.fromarray(((decode_motion2[0,:,:,0]*255+255)/2).astype(np.uint8))
        img.save('check2/decode_motion2_%d.png'%i)
    # features = np.array(features)
    # np.save('features.npy', features)
    # features = np.load('features.npy')
    # km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    # pred_km = km.fit_predict(features)
    # centers = kmeans(features, num_centers=n_clusters, initialization='PLUSPLUS', num_repetitions=10,
    #                max_num_comparisons=100, max_num_iterations=100, algorithm='LLOYD', num_trees=3)
    # pred_km = kmeans_quantize(features, centers)
    # np.save('pred_km.npy', pred_km)
    # sparse_labels = np.eye(n_clusters)[pred_km]
    # sparse_labels = (sparse_labels - 0.5) * 2
    # # pred_km = np.load('pred_km.npy')
    # base_estimizer = SGDClassifier(max_iter=10000, warm_start=True, loss='hinge', early_stopping=True, n_iter_no_change=50, l1_ratio=0)
    # ovr_classifier = OneVsRestClassifier(base_estimizer)
    # ovr_classifier.fit(features, sparse_labels)

    # # clf = LinearSVC(C=1.0, multi_class='ovr', max_iter=100000, loss='hinge')
    # # clf.fit(features, pred_km)
    # dump(ovr_classifier, os.path.join(args.result_directory, 'svm.pickle'))
