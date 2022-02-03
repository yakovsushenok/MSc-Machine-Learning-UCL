import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Concatenate, Dense, Flatten, BatchNormalization, Conv2D, Activation, AveragePooling2D
import numpy as np
import os
import random as rn
from PIL import Image



# Loading the data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

def cutout(image, s, nSquares):
        """
        This function takes in an image and "cuts out" a square out of the image making the region that is "cut out" to be black.
        Args:
                image - the image to which we want to cut out a part
                s - a parameter which is used to determine the interval from which we choose the size of the square we want to cut out from
                nSquares - number of squares we'd like to cut out
        Returns:
                ret - an augmented image with a cut out squared region
        """
        mask_size = tf.random.uniform([1], minval = 0, maxval = s, dtype = tf.dtypes.int32)[0]
        height_of_img, width_of_img, _ = image.shape
        ret = image
        for i in range(nSquares):
                x = np.random.randint(width_of_img)
                y = np.random.randint(height_of_img)
                x1 = np.clip(x - mask_size // 2, 0, width_of_img)
                y1 = np.clip(y - mask_size // 2, 0, height_of_img)
                x2 = np.clip(x + mask_size // 2, 0, width_of_img)
                y2 = np.clip(y + mask_size // 2, 0, height_of_img)
                ret[y1:y2, x1:x2, :] = 0
        return ret

# Applying the cutout to the images in the training set
for image in train_images:
        image = cutout(image, 15, 1)

# Printing out an example of 16 augmented images which are about to be fed into the network
num_images = 16
im = Image.fromarray(tf.concat([train_images[i, ...] for i in range(num_images)], 1).numpy()) 
im.save("cutout.png") 
print('\n---------------cutout.png saved--------------------\n')

# Normalizing the images
train_images = train_images / 255.0
test_images = test_images / 255.0

def comp_function(x_input, numOfFilters): 
        """
        This is the composite function
        Args:
                x_input - the input tensor
                numOfFilters - number of filters
        Returns:
                updated tensor
        """
        x = layers.BatchNormalization()(x_input) 
        x = layers.Activation('relu')(x) 
        x_output = layers.Conv2D(filters = numOfFilters, kernel_size = (3, 3), padding = 'same')(x)

        return x_output

def dense_block(x, numOfFilters, growth_rate):
        """
        This function represents the dense block in the DenseNet architechture. It has 4 convolutional layers.
        Args:
                x - the input tensor
                numOfFilters - number of filters
                growth_rate - growth rate
        Returns:
                updated output tensor and number of filters
        """
        for convolutional_layer in range(4):      
                comp_layer = comp_function(x, growth_rate)
                x = layers.Concatenate()([x, comp_layer]) # each input is a concatenation of all the previous outputs 
                numOfFilters += growth_rate 

        return x, numOfFilters

def transition_layer(inputs, numOfFilters):  
        """
        This is the transition layer
        Args:
                inputs - input tensor
                numOfFilters - number of filters
        Returns:
                updated tensor
        """
        x = layers.BatchNormalization()(inputs)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters = numOfFilters, kernel_size=(1, 1), padding = 'same')(x)
        x = layers.AveragePooling2D(pool_size = (2, 2), padding = 'same')(x)

        return x

##############################################################################################
#                                                                                            #
#                               Building the model                                           #
#                                                                                            #
##############################################################################################
inputs = Input(shape = (32, 32, 3))

# First convolutional layer

x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(inputs)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

# 3 consecutive Dense blocks 
filters = 12
for block in range(2):
        x,filters = dense_block(x, filters, 12) 
        filters = filters * 0.5 # Compression
        x = transition_layer(x, filters) 

x,filters = dense_block(x, filters, 12) 
# Output layer
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.AveragePooling2D(pool_size = (2,2), padding = 'same')(x)
x = layers.Flatten()(x)
x = Dense(units = 64, activation = 'relu')(x)
outputs = Dense(units = 10, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = outputs)
model.summary()

##############################################################################################
#                                                                                            #
#                               Evaluating the model                                         #
#                                                                                            #
##############################################################################################

# Compile model
model.compile(optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy'])

# train 
# We use our test images and labels to get the test set error at every epoch. This does not affect the training of the model (https://stackoverflow.com/questions/46697315/how-does-validation-data-affect-learning-in-keras)
history = model.fit(train_images, train_labels, epochs = 10, validation_data= (test_images, test_labels))

print('\n-------------Training done-------------------\n')

loss_epochs = history.history['val_accuracy']
print('\n----------------Epochs versus accuracy-----------\n')
for i,loss in enumerate(loss_epochs):
        print(f"At epoch {i+1} the accuracy is equal to {loss}")
        print('\n')

# save trained model
model.save('saved_model_tf')

print('\n--------------------Model saved--------------------\n')
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
## inference
num_images = 36
outputs = model.predict(test_images[:num_images, ...] / 255.0)
predicted = tf.argmax(outputs, 1)


print('Getting the ground-truth and predicted labels for 36 images:\n')
for i in range(num_images):
    print(f"Ground-Truth: {class_names[test_labels[i,0]]} ------- Predicted: {class_names[predicted[i]]}")

# example images
im = Image.fromarray(tf.concat([test_images[i, ...] for i in range(num_images)], 1).numpy())
im.save("result.png")

print('\n----------------result.png saved---------------\n')

