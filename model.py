import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



path = "data"

samples = []
with open("{}/driving_log.csv".format(path)) as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:  
    samples.append(line)
    
def process_image(img_path):
    filename = img_path.split("/")[-1]
    filepath = "{}/IMG/{}".format(path, filename) 
    return cv2.imread(filepath)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                correction = 0.1 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction


                img_center = process_image(batch_sample[0])
                img_left = process_image(batch_sample[1])
                img_right = process_image(batch_sample[2])

                images.append(img_center)
                images.append(img_left)
                images.append(img_right)

                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)

                image_flipped = np.fliplr(img_center)
                images.append(image_flipped)

                measurement_flipped = -steering_center
                angles.append(measurement_flipped)
                
                image_flipped = np.fliplr(img_left)
                images.append(image_flipped)

                measurement_flipped = -steering_left
                angles.append(measurement_flipped)
                
                image_flipped = np.fliplr(img_right)
                images.append(image_flipped)

                measurement_flipped = -steering_right
                angles.append(measurement_flipped)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


train_samples, validation_samples = train_test_split(samples, test_size=0.3)

batch_size=32
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= int(len(train_samples)/batch_size), 
                    validation_data=validation_generator, nb_val_samples=int(len(validation_samples)/batch_size), nb_epoch=3)
model.save("model.h5")