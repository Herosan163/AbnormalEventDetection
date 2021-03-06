import numpy as np
import tensorflow as tf
from model import convolutional_auto_encoder
import os
from PIL import Image
import cv2
import argparse
import pickle

tf.compat.v1.enable_eager_execution()

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='ssd')
    return graph

@tf.function
def calc_loss1(model, x):
    encode = model.encode(x)
    decode = model.decode(encode)
    loss = tf.reduce_mean(tf.square(x - decode))
    return loss

def train_step1(model, optimizer, image):
    with tf.GradientTape() as tape:
        loss = calc_loss1(model, image)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def calc_loss2(model, x):
    encode = model.encode(x)
    decode = model.decode(encode)
    loss = tf.reduce_mean(tf.square(x - decode))
    return loss

def train_step2(model, optimizer, image):
    with tf.GradientTape() as tape:
        loss = calc_loss2(model, image)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def calc_loss3(model, x):
    encode = model.encode(x)
    decode = model.decode(encode)
    loss = tf.reduce_mean(tf.square(x - decode))
    return loss

def train_step3(model, optimizer, image):
    with tf.GradientTape() as tape:
        loss = calc_loss3(model, image)
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
    sess = tf.Session(graph=graph)

    train_dir = '../UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/'
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
            if (not ext == '.tif' or int(name[:3]) < 4 or
                not os.path.exists(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])+3)))):
                continue
            img = Image.open(fp)
            img_org = img.copy()
            gray = np.array(img.convert('L'))
            former = Image.open(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])-3))).convert('L')
            later = Image.open(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])+3))).convert('L')
            former = np.array(former)
            later = np.array(later)
            img_rgb = img_org.convert('RGB')
            img_rgb = np.array(img_rgb.resize((640, 640), Image.BILINEAR))
            boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                   feed_dict={x: img_rgb[None]})
            boxes, scores, classes = boxes[0], scores[0], classes[0]
            height, width = gray.shape
            print(fp)
            for i, sc in enumerate(scores):
                if sc > threshold:
                    xmin = int(width * boxes[i][1])
                    ymin = int(height * boxes[i][0])
                    xmax = int(width * boxes[i][3])
                    ymax = int(height * boxes[i][2])
                    gray_copy = gray.copy()
                    former_copy = former.copy()
                    later_copy = later.copy()
                    im = Image.fromarray(gray_copy[ymin:ymax, xmin:xmax])
                    im = im.resize((64, 64), Image.BILINEAR)
                    im = np.array(im)
                    former_copy = Image.fromarray(former_copy[ymin:ymax, xmin:xmax])
                    former_copy = former_copy.resize((64, 64), Image.BILINEAR)
                    former_copy = np.array(former_copy)
                    later_copy = Image.fromarray(later_copy[ymin:ymax, xmin:xmax])
                    later_copy = later_copy.resize((64, 64), Image.BILINEAR)
                    later_copy = np.array(later_copy)
                    data.append(im)
                    data_former.append(former_copy)
                    data_later.append(later_copy)
    return np.array(data), np.array(data_former), np.array(data_later)


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
    data, former_grad, later_grad = create_gradient(data, data_former, data_later)
    data = np.expand_dims(data, axis=-1)
    former_grad = np.expand_dims(former_grad, axis=-1)
    later_grad = np.expand_dims(later_grad, axis=-1)
    data /= 255.
    former_grad /= 255.
    later_grad /= 255.
    train_ds = tf.data.Dataset.from_tensor_slices((data, former_grad, later_grad)).shuffle(shuffle_buffer_size).batch(batch_size)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=data.shape[0]//batch_size * 100,
        decay_rate=0.1,
        staircase=True)
    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=data.shape[0]//batch_size * 100,
        decay_rate=0.1,
        staircase=True)
    lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=data.shape[0]//batch_size * 100,
        decay_rate=0.1,
        staircase=True)
    model_appearance = convolutional_auto_encoder(name='appearance')
    model_motion1 = convolutional_auto_encoder(name='motion1')
    model_motion2 = convolutional_auto_encoder(name='motion2')
    # input_shape = (None, 64, 64, 1)
    # model_appearance.build(input_shape)
    # model_motion1.build(input_shape)
    # model_motion2.build(input_shape)
    # model_appearance.load_weights('result/012/model_appearance_epoch24.h5')
    # model_motion1.load_weights('result/012/model_motion1_epoch24.h5')
    # model_motion2.load_weights('result/012/model_motion2_epoch24.h5')
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule1)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule2)
    
    for epoch in range(args.epoch):
        step = 0
        loss_running = 0.
        loss1_running = 0.
        loss2_running = 0.
        for img, gradient1, gradient2 in train_ds:
            step += 1
            loss = train_step1(model_appearance, optimizer, img)
            loss1 = train_step2(model_motion1, optimizer1, gradient1)
            loss2 = train_step3(model_motion2, optimizer2, gradient2)
            loss_running += loss.numpy()
            loss1_running += loss1.numpy()
            loss2_running += loss2.numpy()
        loss_running /= len(list(train_ds))
        loss1_running /= len(list(train_ds))
        loss2_running /= len(list(train_ds))
        print('epoch', epoch, 'loss', loss_running, 'loss_gradient_former', loss1_running, 'loss_gradient_later', loss2_running)
        with open(os.path.join(args.result_directory, 'log'), 'a') as f:
            f.write('epoch: ' + str(epoch) + ', loss: ' + str(loss_running) + ', loss gradient former: ' +
                    str(loss1_running) + ', loss gradient later: ' + str(loss2_running) + '\n')
        if (epoch + 1) % 25 == 0:
            model_appearance.save_weights(os.path.join(args.result_directory, 'model_appearance_epoch%d.h5' % (epoch)))
            model_motion1.save_weights(os.path.join(args.result_directory, 'model_motion1_epoch%d.h5' % (epoch)))
            model_motion2.save_weights(os.path.join(args.result_directory, 'model_motion2_epoch%d.h5' % (epoch)))
