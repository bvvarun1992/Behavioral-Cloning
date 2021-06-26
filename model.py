import csv
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Reading image paths and steering angles from excel
samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    ## Skipping first line to avoid reading headers ##
    next(reader)
    ##################################################
    for line in reader:
        samples.append(line)
        
# Splitting data for training and validating        
train_samples, validation_samples = train_test_split(samples, test_size=0.15)
    
# Function to yeild generator for training and validating neural network
def generator(samples, batch_size = 32):
    
    while True:
        shuffle(samples)
    
        for offset in range(0, len(samples), batch_size):
            
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for sample in batch_samples:
                for i in range(3):
                    
                    file_name = '/opt/carnd_p3/data/IMG/'+sample[i].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
                    images.append(image)
                    
                    flip_image = np.fliplr(image)
                    images.append(flip_image)

                    if (i==0):
                        angle = float(sample[3])
                        flip_angle = -1.0*float(sample[3])
                    elif (i==1):
                        angle = float(sample[3]) + 0.2
                        flip_angle = -1.0*(float(sample[3]) + 0.2)
                    elif (i==2):
                        angle = float(sample[3]) - 0.2
                        flip_angle = -1.0*(float(sample[3]) - 0.2)
                        
                    angles.append(angle)
                    angles.append(flip_angle)
                    
            x_train = np.array(images)
            y_train = np.array(angles)
            
            yield x_train, y_train
            
# Setting batch size
batch_size = 32

# Creating generators for training and validating
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Building neural network model 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers import Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import Dropout

model = Sequential()
# Normailizing 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Cropping top portion of the images
model.add(Cropping2D(cropping=((70,25), (0,0))))

# 5x5 layer convolutions
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))

# 3x3 layer convolutions
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))


# Fully convolution layer
model.add(Flatten())
model.add(Dense(100, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, 
                    steps_per_epoch = len(train_samples)/batch_size, 
                    validation_data = validation_generator,
                    validation_steps= len(validation_samples)/batch_size,
                    shuffle = True,
                    epochs=5, verbose=1)

model.save('Model_run2.h5')
print('Model saved')
model.summary()




    
