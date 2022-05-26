import gradio as gr #python version greator than 3.7
import matplotlib.pyplot as plt
import os,PIL
from os import environ
from PIL import Image
import numpy as np
np.random.seed(1)

import tensorflow as tf
tf.random.set_seed(1)

from tensorflow.python import keras
from keras import layers,models
import pathlib

#change environment variable
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

suppress_qt_warnings()


os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
#prevent tensorflow from accessing gpu

data_dir = "D:/python project/dataset2/" #in your computer you need to change the path to where you store it

data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))

print("total images count：",image_count)
sunrise = list(data_dir.glob('sunrise/*.jpg'))
img=PIL.Image.open(str(sunrise[0]))
#img.show()


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(20, 10))
for images, labels in train_ds.take(1):
    for i in range(20):
        ax = plt.subplot(5, 10, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        
        plt.axis("off")
    plt.show()
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#prefetching data and get them in cache to accelarate the running speed
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

num_classes = 4
#build convolutional neural network
model = models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255 ),
    
    layers.Conv2D(16, (3, 3), activation='relu'),  # extract the feature of image (edge)  
    layers.AveragePooling2D((2, 2)),               # to get the average value of each partition of the matrix 
    layers.Conv2D(32, (3, 3), activation='relu'),  
    layers.AveragePooling2D((2, 2)),               
    layers.Conv2D(64, (3, 3), activation='relu'),  
    layers.Dropout(0.3),  
    
    layers.Flatten(),                       
    layers.Dense(128, activation='relu'),   
    layers.Dense(num_classes)               
])

model.build(input_shape=(0,img_height, img_width, 3))
#model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, #set optimizer to optimize the model
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #use loss function to measure the accuracy 
              metrics=['accuracy']) #moniter the ratio of the image that be classified correctly



epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#evaluate the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

def predict_image(img):
    class_names=['cloudy','rain','shine','sunrise']
    img_shape=img.reshape(-1,180,180,3)
    prediction=model.predict(img_shape)[0]
    return {class_names[i]: float(prediction[i]) for i in range(4)}

images=gr.inputs.Image(shape=(180,180))
labels=gr.outputs.Label(num_top_classes=4)
gr.Interface(fn=predict_image, inputs=images, outputs=labels).launch(share=True)
