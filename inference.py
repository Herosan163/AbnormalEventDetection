import numpy as np
import tensorflow as tf
from model import convolutional_auto_encoder
import os
from PIL import Image
import cv2
import argparse
import pickle
from joblib import dump, load
from sklearn import metrics
from scipy import signal

tf.compat.v1.enable_eager_execution()

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='ssd')
    return graph

def svm_predict(svm, feature):
    scores = svm.decision_function(feature[None])
    return -np.amax(np.array(scores))

def smooth(x,window_len=11,window='hanning'):
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = eval('np.'+window+'(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    ssd_threshold = 0.4
    frozen_model_filepath = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
    graph = load_graph(frozen_model_filepath)

    x = graph.get_tensor_by_name('ssd/image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('ssd/detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('ssd/detection_scores:0')
    detection_classes = graph.get_tensor_by_name('ssd/detection_classes:0')
    num_detections = graph.get_tensor_by_name('ssd/num_detections:0')
    sess = tf.Session(graph=graph)
    
    dummpy = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model_appearance = convolutional_auto_encoder()
    model_appearance.compile(loss=tf.keras.losses.MSE,
                             optimizer=tf.keras.optimizers.Adam())
    model_appearance.train_on_batch(dummpy, dummpy)
    model_motion1 = convolutional_auto_encoder()
    model_motion1.compile(loss=tf.keras.losses.MSE,
                          optimizer=tf.keras.optimizers.Adam())
    model_motion1.train_on_batch(dummpy, dummpy)
    model_motion2 = convolutional_auto_encoder()
    model_motion2.compile(loss=tf.keras.losses.MSE,
                          optimizer=tf.keras.optimizers.Adam())
    model_motion2.train_on_batch(dummpy, dummpy)
    model_appearance.load_weights(os.path.join(args.result_directory, 'model_appearance_best_loss.hdf5'))
    model_motion1.load_weights(os.path.join(args.result_directory, 'model_motion1_best_loss.hdf5'))
    model_motion2.load_weights(os.path.join(args.result_directory, 'model_motion2_best_loss.hdf5'))

    svm = load(os.path.join(args.result_directory, 'svm.pickle'))

    test_dir = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'
    result_dir = 'detected_images'
    frame_scores = []
    frame_gt = []
    for dirname in sorted(os.listdir(test_dir)):
        basename = os.path.basename(dirname)
        if not basename[:4] == 'Test':
            continue
        for fn in sorted(os.listdir(os.path.join(test_dir, dirname))):
            fp = os.path.join(test_dir, dirname, fn)
            name, ext = os.path.splitext(fn)
            if (not ext == '.tif' or int(name[:3]) < 3 or
                not os.path.exists(os.path.join(test_dir, dirname, '%03d.tif' % (int(name[:3])+2)))):
                continue
            im = Image.open(fp)
            im_former = Image.open(os.path.join(test_dir, dirname, '%03d.tif' % (int(name[:3])-2))).convert('L')
            im_later = Image.open(os.path.join(test_dir, dirname, '%03d.tif' % (int(name[:3])+2))).convert('L')
            if dirname in ['Test006', 'Test009', 'Test012']:
                gt = Image.open(os.path.join(test_dir, dirname + '_gt', 'frame%03d.bmp' % (int(name[:3]))))
            else:
                gt = Image.open(os.path.join(test_dir, dirname + '_gt', '%03d.bmp' % (int(name[:3]))))                
            im_former = np.array(im_former)
            im_later = np.array(im_later)
            gt = np.array(gt)
            if len(np.where(gt != 0)[0]) == 0:
                frame_class = 0
            else:
                frame_class = 1
            frame_gt.append(frame_class)
            im_org = im.copy()
            im_org = np.array(im_org.convert('L'))
            img = im.convert('RGB')
            img = np.array(img.resize((600, 600), Image.LANCZOS))
            boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                   feed_dict={x: img[None]})
            boxes, scores, classes = boxes[0], scores[0], classes[0]
            height, width = im_org.shape
            print(fp)
            frame_score = -np.inf
            for i, sc in enumerate(scores):
                if sc > ssd_threshold:
                    xmin = int(width * boxes[i][1])
                    ymin = int(height * boxes[i][0])
                    xmax = int(width * boxes[i][3])
                    ymax = int(height * boxes[i][2])
                    im_copy = im_org.copy()
                    im_copy_former = im_former.copy()
                    im_copy_later = im_later.copy()
                    im = Image.fromarray(im_copy[ymin:ymax, xmin:xmax])
                    im = np.array(im.resize((64, 64), Image.LANCZOS))
                    im_copy_former = Image.fromarray(im_copy_former[ymin:ymax, xmin:xmax])
                    im_copy_former = im_copy_former.resize((64, 64), Image.LANCZOS)
                    im_copy_former = np.array(im_copy_former)
                    im_copy_later = Image.fromarray(im_copy_later[ymin:ymax, xmin:xmax])
                    im_copy_later = im_copy_later.resize((64, 64), Image.LANCZOS)
                    im_copy_later = np.array(im_copy_later)
                    img = tf.cast(tf.identity(np.expand_dims(im, axis=-1)), tf.float32)
                    former = tf.cast(tf.identity(np.expand_dims(im_copy_former, axis=-1)), tf.float32)
                    later = tf.cast(tf.identity(np.expand_dims(im_copy_later, axis=-1)), tf.float32)
                    encode_appearance = model_appearance.encode(img[None] / 255.0).numpy()
                    encode_motion1 = model_motion1.encode((img - former)[None] / 255.0).numpy()
                    encode_motion2 = model_motion2.encode((later - img)[None] / 255.0).numpy()
                    feature = np.r_[encode_appearance.flatten(), encode_motion1.flatten(), encode_motion2.flatten()]
                    score = svm_predict(svm, feature)
                    if frame_score < score:
                        frame_score = score
            frame_scores.append(frame_score)
            print(frame_class, frame_score)
    frame_scores = np.array(frame_scores)
    # np.save('frame_scores.npy', frame_scores)
    frame_gt = np.array(frame_gt)
    # np.save('frame_gt.npy', frame_gt)
    frame_scores = np.load('frame_scores.npy')
    # frame_gt = np.load('frame_gt.npy')
    print(frame_scores.shape, frame_gt.shape)
    # frame_scores = smooth(frame_scores)
    fpr, tpr, thresholds = metrics.roc_curve(frame_gt, frame_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('AUC: ', auc)
