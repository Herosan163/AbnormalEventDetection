import numpy as np
import tensorflow as tf
from model import convolutional_auto_encoder
import os
from PIL import Image
import cv2
import argparse
import pickle


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='ssd')
    return graph


def create_dataset(threshold):

    frozen_model_filepath = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
    graph = load_graph(frozen_model_filepath)

    x = graph.get_tensor_by_name('ssd/image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('ssd/detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('ssd/detection_scores:0')
    detection_classes = graph.get_tensor_by_name('ssd/detection_classes:0')
    num_detections = graph.get_tensor_by_name('ssd/num_detections:0')
    sess = tf.Session(graph=graph)

    train_dir = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/'
    result_dir = 'detected_images'
    data = []
    data_former = []
    data_later = []
    for dirname in sorted(os.listdir(train_dir)):
        basename = os.path.basename(dirname)
        if not basename[:5] == 'Train':
            continue
        for fn in sorted(os.listdir(os.path.join(train_dir, dirname))):
            fp = os.path.join(train_dir, dirname, fn)
            name, ext = os.path.splitext(fn)
            if (not ext == '.tif' or int(name[:3]) < 3 or
                not os.path.exists(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])+2)))):
                continue
            im = Image.open(fp)
            im_former = Image.open(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])-2))).convert('L')
            im_later = Image.open(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])+2))).convert('L')
            im_former = np.array(im_former)
            im_later = np.array(im_later)
            im_org = im.copy()
            im_org = np.array(im_org.convert('L'))
            img = im.convert('RGB')
            img = np.array(img.resize((600, 600), Image.LANCZOS))
            boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                   feed_dict={x: img[None]})
            boxes, scores, classes = boxes[0], scores[0], classes[0]
            height, width = im_org.shape
            print(fp)
            for i, sc in enumerate(scores):
                if sc > threshold:
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
                    data.append(im)
                    data_former.append(im_copy_former)
                    data_later.append(im_copy_later)
    return np.array(data), np.array(data_former), np.array(data_later)


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    if not os.path.exists(args.result_directory):
        os.mkdir(args.result_directory)
    threshold = 0.5
    batch_size = 64
    shuffle_buffer_size = 100
    # data, data_former, data_later = create_dataset(threshold)
    # np.save('data.npy', data)
    # np.save('data_former.npy', data_former)
    # np.save('data_later.npy', data_later)
    data = np.load('data.npy')
    data_former = np.load('data_former.npy')
    data_later = np.load('data_later.npy')
    data = np.expand_dims(data, axis=-1).astype(np.float32)
    data_former = np.expand_dims(data_former, axis=-1).astype(np.float32)
    data_later = np.expand_dims(data_later, axis=-1).astype(np.float32)
    data /= 255.
    data_former /= 255.
    data_later /= 255.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=data.shape[0]//batch_size * 100,
        decay_rate=0.1,
        staircase=True)
    dummpy = np.zeros((1, 64, 64, 1), dtype=np.float32)
    model_appearance = convolutional_auto_encoder()
    model_motion1 = convolutional_auto_encoder()
    model_motion2 = convolutional_auto_encoder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer3 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model_appearance.compile(optimizer=optimizer,
                             loss='mse')
    model_appearance.fit(data, data, epochs=args.epoch, batch_size=batch_size)
    model_appearance.save_weights(os.path.join(args.result_directory, 'model_appearance_best_loss.hdf5'))
    model_motion1.compile(optimizer=optimizer2,
                          loss='mse')
    model_motion1.fit(data_former, data_former, epochs=args.epoch, batch_size=batch_size)
    model_motion1.save_weights(os.path.join(args.result_directory, 'model_motion1_best_loss.hdf5'))
    model_motion2.compile(optimizer=optimizer3,
                          loss='mse')
    model_motion2.fit(data_later, data_later, epochs=args.epoch, batch_size=batch_size)
    model_motion2.save_weights(os.path.join(args.result_directory, 'model_motion2_best_loss.hdf5'))
