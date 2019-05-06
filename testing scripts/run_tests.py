import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
import pandas as pd
import numpy as np
import csv
from keras.optimizers import Adam
import sys


test_dir = '../datasets/server/test'
dictionary = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

model_name = sys.argv[1]+'.h5'
model = models.load_model(model_name)
model.compile(loss='categorical_crossentropy',
optimizer=Adam(lr=0.00001), metrics=['acc'])

X = pd.read_csv('./datasets/server/tests.csv', sep=',', skiprows=1, names=['file_name', 'tag'])

fnames = X['file_name']
tag = X['tag']

test_file_name= 'test_'+sys.argv[1]+'.csv'
for i in range(fnames.shape[0]):
    src = test_dir+'/'+fnames[i]
    test_image = image.load_img(src, target_size = (200, 200))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image) #confiabilidad
    classes = model.predict_classes(test_image)
    print("'"+dictionary[classes]+"'")
    print("'"+tag[i]+"'")
    is_correct = dictionary[classes] is tag[i]
    with open(test_file_name, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([fnames[i], tag[i] , dictionary[classes], is_correct])
