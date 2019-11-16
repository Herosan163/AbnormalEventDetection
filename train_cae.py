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


def train_step(model, train_loss, optimizer, image):
    with tf.GradientTape() as tape:
        loss = model.calc_loss(image)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def create_dataset(threshold):

    frozen_model_filepath = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
    graph = load_graph(frozen_model_filepath)

    x = graph.get_tensor_by_name('ssd/image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('ssd/detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('ssd/detection_scores:0')
    detection_classes = graph.get_tensor_by_name('ssd/detection_classes:0')
    num_detections = graph.get_tensor_by_name('ssd/num_detections:0')

    train_dir = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/'
    result_dir = 'detected_images'
    data = []
    gradient_former = []
    gradient_later = []
    with tf.Session(graph=graph) as sess:
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
                        im_grad = cv2.Laplacian(im.copy(), cv2.CV_32F, ksize=3)
                        former_grad = cv2.Laplacian(im_copy_former.copy(), cv2.CV_32F, ksize=3)
                        later_grad = cv2.Laplacian(im_copy_later.copy(), cv2.CV_32F, ksize=3)
                        gradient_former.append(im_grad + former_grad)
                        gradient_later.append(later_grad + im_grad)
    return np.array(data), np.array(gradient_former), np.array(gradient_later)


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float)
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
    # data, grad_former, grad_later = create_dataset(threshold)
    # np.save('data.npy', data)
    # np.save('grad_former.npy', grad_former)
    # np.save('grad_later.npy', grad_later)
    data = np.load('data.npy')
    grad_former = np.load('grad_former.npy')
    grad_later = np.load('grad_later.npy')
    data = np.expand_dims(data, axis=-1)
    grad_former = np.expand_dims(grad_former, axis=-1)
    grad_later = np.expand_dims(grad_later, axis=-1)
    train_ds = tf.data.Dataset.from_tensor_slices((data, grad_former, grad_later)).shuffle(shuffle_buffer_size).batch(batch_size)
    model_appearance = convolutional_auto_encoder()
    model_motion1 = convolutional_auto_encoder()
    model_motion2 = convolutional_auto_encoder()
    loss_object = tf.keras.losses.MSE
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    loss_object2 = tf.keras.losses.MSE
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_loss2 = tf.keras.metrics.Mean(name='train_loss')
    loss_object3 = tf.keras.losses.MSE
    optimizer3 = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_loss3 = tf.keras.metrics.Mean(name='train_loss')

    EPOCHS = args.epoch
    delta = 2
    best_loss1 = 10000000
    best_loss2 = 10000000
    best_loss3 = 10000000
    for epoch in range(EPOCHS):
        step = 0
        loss1_running = 0.
        loss2_running = 0.
        loss3_running = 0.
        for img, gradient1, gradient2 in train_ds:
            step += 1
            loss1 = train_step(model_appearance, train_loss, optimizer, tf.identity(img))
            loss2 = train_step(model_motion1, train_loss2, optimizer2, tf.identity(gradient1))
            loss3 = train_step(model_motion2, train_loss3, optimizer3, tf.identity(gradient2))
            loss1_running += loss1.numpy()
            loss2_running += loss2.numpy()
            loss3_running += loss3.numpy()
        loss1_running /= len(list(train_ds))
        loss2_running /= len(list(train_ds))
        loss3_running /= len(list(train_ds))
        print('loss1_running', loss1_running, 'loss2_running', loss2_running, 'loss3_running', loss3_running)
        with open(os.path.join(args.result_directory, 'log'), 'a') as f:
            f.write('loss1_running: ' + str(loss1_running) + ', loss2_running: ' + str(loss2_running) + ', loss3_running: ' + str(loss3_running) + '\n')
        if best_loss1 > loss1_running:
            best_loss1 = loss1_running
            model_appearance.save_weights(os.path.join(args.result_directory, 'model_appearance_best_loss.h5'))
        if best_loss2 > loss2_running:
            best_loss2 = loss2_running
            model_motion1.save_weights(os.path.join(args.result_directory, 'model_motion1_best_loss.h5'))
        if best_loss3 > loss3_running:
            best_loss3 = loss3_running
            model_motion2.save_weights(os.path.join(args.result_directory, 'model_motion2_best_loss.h5'))
