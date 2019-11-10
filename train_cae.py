import numpy as np
import tensorflow as tf
from model import convolutional_auto_encoder

@tf.function
def train_step(model, train_loss, optimizer, image, label):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = train_loss(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


def create_dataset(image_dir, delta):

    i = 0
    while os.path.exists():
        
        i += 1


if __name__ == '__main__':
    model_appearance = convolutional_auto_encoder()
    model_motion1 = convolutional_auto_encoder()
    model_motion2 = convolutional_auto_encoder()
    loss_object = tf.keras.losses.MSE()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    EPOCHS = 5
    delta = 2

    for epoch in range(EPOCHS):
        for image, label in train_ds:
            train_step(image, label)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))
