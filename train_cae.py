import numpy as np
import tensorflow as tf
from model import convolutional_auto_encoder
import os
from PIL import Image

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='ssd')
    return graph


@tf.function
def train_step(model, train_loss, optimizer, image):
    with tf.GradientTape() as tape:
        encode, decode = model(image)
        loss = tf.reduce_mean(tf.square(image - decode))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


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
    data_before = []
    data_after = []
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
                im_before = Image.open(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])-2))).convert('L')
                im_after = Image.open(os.path.join(train_dir, dirname, '%03d.tif' % (int(name[:3])+2))).convert('L')
                im_before = np.array(im_before)
                im_after = np.array(im_after)
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
                        im_copy_before = im_before.copy()
                        im_copy_after = im_after.copy()
                        im = Image.fromarray(im_copy[ymin:ymax, xmin:xmax])
                        im = np.array(im.resize((64, 64), Image.LANCZOS))
                        im_copy_before = Image.fromarray(im_copy_before[ymin:ymax, xmin:xmax])
                        im_copy_before = im_copy_before.resize((64, 64), Image.LANCZOS)
                        im_copy_before = np.array(im_copy_before)
                        im_copy_after = Image.fromarray(im_copy_after[ymin:ymax, xmin:xmax])
                        im_copy_after = im_copy_after.resize((64, 64), Image.LANCZOS)
                        im_copy_after = np.array(im_copy_after)
                        data.append(im)
                        data_before.append(im_copy_before)
                        data_after.append(im_copy_after)
    return np.array(data), np.array(data_before), np.array(data_after)


if __name__ == '__main__':

    threshold = 0.5
    batch_size = 64
    shuffle_buffer_size = 100
    # data, data_before, data_after = create_dataset(threshold)
    # np.save('data.npy', data)
    # np.save('data_before.npy', data_before)
    # np.save('data_after.npy', data_after)
    data = np.load('data.npy')
    data_before = np.load('data_before.npy')
    data_after = np.load('data_after.npy')
    train_ds = tf.data.Dataset.from_tensor_slices((data, data_before, data_after)).shuffle(shuffle_buffer_size).batch(batch_size)
    model_appearance = convolutional_auto_encoder()
    model_motion1 = convolutional_auto_encoder()
    model_motion2 = convolutional_auto_encoder()
    loss_object = tf.keras.losses.MSE()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    loss_object2 = tf.keras.losses.MSE()
    optimizer2 = tf.keras.optimizers.Adam()
    train_loss2 = tf.keras.metrics.Mean(name='train_loss')
    loss_object3 = tf.keras.losses.MSE()
    optimizer3 = tf.keras.optimizers.Adam()
    train_loss3 = tf.keras.metrics.Mean(name='train_loss')

    EPOCHS = 5
    delta = 2

    for epoch in range(EPOCHS):
        for img, gradient1, gradient2 in train_ds:
            train_step(model_appearance, train_loss, optimizer, img.copy())            
            train_step(model_motion1, train_loss2, optimizer2, img.copy())            
            train_step(model_motion2, train_loss3, optimizer3, img.copy())            

        template = 'Epoch {}, Loss: {}'
        print (template.format(epoch+1, train_loss.result()))
        template = 'Epoch {}, Loss: {}'
        print (template.format(epoch+1, train_loss2.result()))
        template = 'Epoch {}, Loss: {}'
        print (template.format(epoch+1, train_loss3.result()))
