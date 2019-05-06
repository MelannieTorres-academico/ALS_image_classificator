import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import Adam
from keras import layers
from keras import models
from keras.applications import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# Get CUDA session
def get_session(gpu_fraction=0.3):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
  return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Call function to only use a small part of the GPU and leave space for others $
KTF.set_session(get_session())


train_dir='../datasets/server/train'
validation_dir = '../datasets/server/validation'


train_datagen = ImageDataGenerator( horizontal_flip=True, width_shift_range=0.2,
height_shift_range=0.2, shear_range=0.2,
zoom_range=0.2,
fill_mode='nearest')

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory( train_dir, target_size=(200, 200), batch_size=20,class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(200, 200), batch_size=20, class_mode='categorical')

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
conv_base.trainable = False

#Transfer Learning
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(29, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=500, epochs=15, validation_data=validation_generator, validation_steps=50)


model.save('modelB.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('figureB_accuracy.png')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('figureB_loss.png')
