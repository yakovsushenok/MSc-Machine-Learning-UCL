import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Concatenate, Dense, Flatten, BatchNormalization, Conv2D, Activation, AveragePooling2D
import numpy as np
import os
import random as rn
from PIL import Image
import time


# Inidicating choice
print("\nI choose \"Difference between training with and without the Cutout data augmentation algorithm implemented in Task 2\".\n ")
# Loading the data into development and holdout set
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
def cutout(image, s, nSquares):
        """
        This function takes in an image and "cuts out" a square out of the image making it black.
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
            ret[y1:y2, x1:x2, :] = 0.0
        return ret
# Getting a dataset with cutout images
print("\nCurrently augmenting the images which are in the development set.....\n")
train_images_cutout = train_images
for image in train_images_cutout:
  image = cutout(np.array(image).astype(np.float32), 15, 1)
print("Finished augmenting the images\n ")
# Normalizing
train_images, test_images, train_images_cutout = np.array(train_images).astype(np.float32)/255.0, np.array(test_images).astype(np.float32)/255.0, np.array(train_images_cutout).astype(np.float32)/255.0
##############################################################################################
#                                                                                            #
#                  We will first start with the model with augmented data                    #
#                                                                                            #
##############################################################################################
print("\nWe will first start with the model with augmented data\n")
class TimeHistory(tf.keras.callbacks.Callback):
    """
    This class is used for recording the time taken in each epoch
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
# Shuffling the development set data
perm = np.random.permutation(len(train_images)) 
train_images_shuffled = train_images[perm] 
train_images_cutout_shuffled = train_images_cutout[perm]
train_labels_shuffled = train_labels[perm] 
# Splitting into 3 folds
s1, s2 = np.int(np.around(0.33*len(train_images))), np.int(np.around(0.66*len(train_images)))
fold1_images, fold2_images, fold3_images = train_images_shuffled[:s1, :, :], train_images_shuffled[s1:s2, :, :], train_images_shuffled[s2:, :, :]
fold1_images_cutout, fold2_images_cutout, fold3_images_cutout = train_images_cutout_shuffled[:s1, :, :], train_images_cutout_shuffled[s1:s2, :, :], train_images_cutout_shuffled[s2:, :, :]
fold1_labels, fold2_labels, fold3_labels = train_labels_shuffled[:s1, :], train_labels_shuffled[s1:s2, :], train_labels_shuffled[s2:, :]
fold_dataset_images = [fold1_images, fold2_images, fold3_images]
fold_dataset_images_cutout = [fold1_images_cutout, fold2_images_cutout, fold3_images_cutout]
fold_dataset_labels = [fold1_labels, fold2_labels, fold3_labels]
# Print data set summary every time the random split is done.
print('\n-----------------------------------Dataset Summary-------------------------------------\n')
print(f'Whole development set number of samples: {train_images.shape[0]}\n')
print("Distribution of labels in the development set")
unique, counts = np.unique(train_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(train_labels)), decimals= 4)} % of the labels")
print(f'\n\nFirst (out of 3) partition  number of samples: {fold1_images.shape[0]}')
print("Distribution of labels in the first partition")
unique, counts = np.unique(fold1_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(fold1_labels)), decimals= 4)} % of the labels")
print(f'\n\nSecond partition  number of samples: {fold2_images.shape[0]}')
print("Distribution of labels in the second partition")
unique, counts = np.unique(fold2_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(fold2_labels)), decimals= 4)} % of the labels")
print(f'\n\nThird partition number of samples: {fold3_images.shape[0]}')
print("Distribution of labels in the third partition")
unique, counts = np.unique(fold3_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(fold3_labels)), decimals= 4)} % of the labels")
print("\n\n")
# Initializing the array for summary statistics of training and validation
avg_validation_loss0 = []
avg_validation_accuracy0 = []
avg_validation_bnentropy0 = []
avg_training_loss = []
avg_training_accuracy = []
avg_training_bnentropy = []
avg_time_per_epoch = []
avg_time_per_fold_training = []
# Starting the 3-fold cross-validation scheme
start_time_training = time.time()
for fold in range(3): 
    print(f"Current validation set is partition number {fold+1}")
    print(f"Current training set is the set of partitions numbered {[j+1 for j,x in enumerate(fold_dataset_images) if j!=fold][0]} and {[j+1 for j,x in enumerate(fold_dataset_images) if j!=fold][1]}")
    # Setting up the partitions. The training set is the augmented data whilst the validation set is non-augmented data. 
    train_fold_images, train_fold_labels =  [x for j,x in enumerate(fold_dataset_images_cutout) if j!=fold], [x for j,x in enumerate(fold_dataset_labels) if j!=fold]
    train_images_current, train_labels_current = tf.concat(train_fold_images, axis = 0), tf.concat(train_fold_labels, axis = 0)
    val_images_current, val_labels_current = fold_dataset_images[fold], fold_dataset_labels[fold]    
    # Building Model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy',tf.keras.metrics.BinaryCrossentropy()])
    time_callback = TimeHistory()
    start_time_fold = time.time()
    history = model.fit(train_images_current, train_labels_current, validation_data = (val_images_current, val_labels_current), epochs=10, callbacks = [time_callback])
    time_fold = time.time() - start_time_fold
    print("\n-----------------------Validation Set Statistics--------------------------")
    print(f"Loss: {np.round(history.history['val_loss'][-1], decimals = 4)}, Accuracy: {np.round(history.history['val_accuracy'][-1], decimals = 4)}, Binary Cross Entropy: {np.round(history.history['val_binary_crossentropy'][-1], decimals = 4)}")
    avg_validation_loss0.append(history.history['val_loss'][-1])
    avg_validation_accuracy0.append(history.history['val_accuracy'][-1])
    avg_validation_bnentropy0.append(history.history['val_binary_crossentropy'][-1])
    print("\n-----------------------Training Set Statistics--------------------------")
    print(f"Loss: {np.round(history.history['loss'][-1], decimals = 4)}, Accuracy: {np.round(history.history['accuracy'][-1], decimals = 4)}, Binary Cross Entropy: {np.round(history.history['binary_crossentropy'][-1], decimals = 4)}")
    avg_training_loss.append(history.history['loss'][-1])
    avg_training_accuracy.append(history.history['accuracy'][-1])
    avg_training_bnentropy.append(history.history['binary_crossentropy'][-1])
    print("\n-------------------------Timing Statistics---------------------------------")
    tr = 1
    epoch_time = []
    for t in time_callback.times:
        print(f"Time taken for epoch {tr} is {np.round(t, decimals = 4)} s")
        epoch_time.append(t)
        tr += 1
    avg_time_per_epoch.append(np.mean(epoch_time))
    print(f"\n Total Time taken for training for current train-validation permutation: {np.round(time_fold, decimals = 4)} s\n\n\n")  
    avg_time_per_fold_training.append(time_fold) 
# Summary Statistics for the whole 3-fold cross-validation scheme
print("------------------------\nSummary Statistics for the whole 3-fold cross-validation scheme-------------------------------------\n")
# Times
print("-----------------------------------Timing-------------------------------------\n")
end_time_training =  time.time() - start_time_training 
print(f"\nTotal time taken to Implement the 3-fold cross-validation scheme: {np.round(end_time_training, decimals = 4)} s, Average time taken per epoch: {np.round(np.mean(avg_time_per_epoch), decimals = 4)} s, Average time for training and validation: {np.round(np.mean(avg_time_per_fold_training), decimals = 4)} s")
# Validation Set Statistics
print("\n-----------------------------------Validation Set Statistics-------------------------------------\n")
print(f"Average Validation Loss: {np.round(np.mean(avg_validation_loss0), decimals = 4)}, Average Validation Accuracy: {np.round(np.mean(avg_validation_accuracy0), decimals = 4)}, Average Validation Binary Cross Entropy: {np.round(np.mean(avg_validation_bnentropy0), decimals = 4)}")
# Training Set Statistics
print("\n-----------------------------------Training Set Statistics-------------------------------------\n")
print(f"Average Training Loss: {np.round(np.mean(avg_training_loss), decimals = 4)}, Average Training Accuracy: {np.round(np.mean(avg_training_accuracy), decimals = 4)}, Average Training Binary Cross Entropy: {np.round(np.mean(avg_training_bnentropy), decimals = 4)}\n\n\n\n\n\n\n")
##############################################################################################
#                                                                                            #
#                  We will now implement the same model but with non-augmented data          #
#                                                                                            #
##############################################################################################
##############################################################################################
#                                                                                            #
#                  NOTE: Same code just without the data augmentation loop                   #
#                                                                                            #
##############################################################################################
print("\nWe will now implement the same thing but with non-augmented data\n")
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# train_images = train_images / 255.0
# test_images = test_images / 255.0
# Shuffling the development set data
perm = np.random.permutation(len(train_images)) 
train_images_shuffled = train_images[perm] 
train_labels_shuffled = train_labels[perm] 
# Splitting into 3 folds
s1, s2 = np.int(np.around(0.33*len(train_images))), np.int(np.around(0.66*len(train_images)))
fold1_images, fold2_images, fold3_images = train_images_shuffled[:s1, :, :], train_images_shuffled[s1:s2, :, :], train_images_shuffled[s2:, :, :]
fold1_labels, fold2_labels, fold3_labels = train_labels_shuffled[:s1, :], train_labels_shuffled[s1:s2, :], train_labels_shuffled[s2:, :]
fold_dataset_images = [fold1_images, fold2_images, fold3_images]
fold_dataset_labels = [fold1_labels, fold2_labels, fold3_labels]
# Print data set summary every time the random split is done.
print('\n-----------------------------------Dataset Summary-------------------------------------\n')
print(f'Whole development set number of samples: {train_images.shape[0]}\n')
print("Distribution of labels in the development set")
unique, counts = np.unique(train_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(train_labels)), decimals= 4)} % of the labels")
print(f'\n\nFirst (out of 3) partition  number of samples: {fold1_images.shape[0]}')
print("Distribution of labels in the first partition")
unique, counts = np.unique(fold1_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(fold1_labels)), decimals= 4)} % of the labels")
print(f'\n\nSecond partition  number of samples: {fold2_images.shape[0]}')
print("Distribution of labels in the second partition")
unique, counts = np.unique(fold2_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(fold2_labels)), decimals= 4)} % of the labels")
print(f'\n\nThird partition number of samples: {fold3_images.shape[0]}')
print("Distribution of labels in the third partition")
unique, counts = np.unique(fold3_labels, return_counts=True)
for i in unique:
    print(f"The class {class_names[i]} consists of {np.round(100*(counts[i]/len(fold3_labels)), decimals= 4)} % of the labels")
print("\n\n")
avg_validation_loss1 = []
avg_validation_accuracy1 = []
avg_validation_bnentropy1 = []
avg_training_loss = []
avg_training_accuracy = []
avg_training_bnentropy = []
avg_time_per_epoch = []
avg_time_per_fold_training = []
start_time_training = time.time()
for fold in range(3): 
    print(f"Current validation set is partition number {fold+1}")
    print(f"Current training set is the set of partitions numbered {[j+1 for j,x in enumerate(fold_dataset_images) if j!=fold][0]} and {[j+1 for j,x in enumerate(fold_dataset_images) if j!=fold][1]}")
    train_fold_images, train_fold_labels =  [x for j,x in enumerate(fold_dataset_images) if j!=fold], [x for j,x in enumerate(fold_dataset_labels) if j!=fold]
    train_images_current, train_labels_current = tf.concat(train_fold_images, axis = 0), tf.concat(train_fold_labels, axis = 0)
    val_images_current, val_labels_current = fold_dataset_images[fold], fold_dataset_labels[fold]
    # Building Model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    time_callback = TimeHistory()
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy',tf.keras.metrics.BinaryCrossentropy()])
    start_time_fold = time.time()
    history = model.fit(train_images_current, train_labels_current, validation_data = (val_images_current, val_labels_current), epochs=10, callbacks = [time_callback])
    time_fold = time.time() - start_time_fold
    
    print("\n-----------------------Validation Set Statistics--------------------------")
    print(f"Loss: {np.round(history.history['val_loss'][-1], decimals = 4)}, Accuracy: {np.round(history.history['val_accuracy'][-1], decimals = 4)}, Binary Cross Entropy: {np.round(history.history['val_binary_crossentropy'][-1], decimals = 4)}")
    avg_validation_loss1.append(history.history['val_loss'][-1])
    avg_validation_accuracy1.append(history.history['val_accuracy'][-1])
    avg_validation_bnentropy1.append(history.history['val_binary_crossentropy'][-1])
    
    print("\n-----------------------Training Set Statistics--------------------------")
    print(f"Loss: {np.round(history.history['loss'][-1], decimals = 4)}, Accuracy: {np.round(history.history['accuracy'][-1], decimals = 4)}, Binary Cross Entropy: {np.round(history.history['binary_crossentropy'][-1], decimals = 4)}")
    avg_training_loss.append(history.history['loss'][-1])
    avg_training_accuracy.append(history.history['accuracy'][-1])
    avg_training_bnentropy.append(history.history['binary_crossentropy'][-1])
    print("\n-------------------------Timing Statistics---------------------------------")
    tr = 1
    epoch_time = []
    for t in time_callback.times:
        print(f"Time taken for epoch {tr} is {np.round(t, decimals = 4)} s")
        epoch_time.append(t)
        tr+=1
    avg_time_per_epoch.append(np.mean(epoch_time))
    print(f"\n Total Time taken for training for current train-validation permutation: {np.round(time_fold, decimals = 4)} s\n\n\n")  
    avg_time_per_fold_training.append(time_fold) 


print("------------------------\nSummary Statistics for the whole 3-fold cross-validation scheme-------------------------------------\n")
print("-----------------------------------Timing-------------------------------------\n")
end_time_training =  time.time() - start_time_training 
print(f"\nTotal time taken to Implement the 3-fold cross-validation scheme: {np.round(end_time_training, decimals = 4)} s, Average time taken per epoch: {np.round(np.mean(avg_time_per_epoch), decimals = 4)} s, Average time for training and validation: {np.round(np.mean(avg_time_per_fold_training), decimals = 4)} s")

print("\n-----------------------------------Validation Set Statistics-------------------------------------\n")
print(f"Average Validation Loss: {np.round(np.mean(avg_validation_loss1), decimals = 4)}, Average Validation Accuracy: {np.round(np.mean(avg_validation_accuracy1), decimals = 4)}, Average Validation Binary Cross Entropy: {np.round(np.mean(avg_validation_bnentropy1), decimals = 4)}")


print("\n-----------------------------------Training Set Statistics-------------------------------------\n")
print(f"Average Training Loss: {np.round(np.mean(avg_training_loss), decimals = 4)}, Average Training Accuracy: {np.round(np.mean(avg_training_accuracy), decimals = 4)}, Average Training Binary Cross Entropy: {np.round(np.mean(avg_training_bnentropy), decimals = 4)}\n\n\n\n\n\n\n")
##############################################################################################
#                                                                                            #
#    Train two further models using the entire development set and save the trained models.  #
#                                                                                            #
##############################################################################################
print("Train two further models using the entire development set and save the trained models.\n")
print("First model will be the one with augmented data.")
# Building Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# Compiling the model
model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy',tf.keras.metrics.BinaryCrossentropy()])
# Fitting
history = model.fit(train_images_cutout, train_labels, validation_data = (test_images, test_labels), epochs=10, callbacks = [time_callback])
# save trained model
model.save('saved_model_cutout_tf')
print('\nModel with cutout saved.\n')

cutout_model_loss = history.history['val_loss'][-1]
cutout_model_acc = history.history['val_accuracy'][-1]

print("\n-----------------------Hold Out Test Set Statistics for model with augmented data--------------------------")
print(f"Loss: {history.history['val_loss'][-1]}, Accuracy: {history.history['val_accuracy'][-1]}, Binary Cross Entropy: {history.history['val_binary_crossentropy'][-1]}\n\n\n\n")

########################################## Comparing the results with those obtained during cross-validation #####################################################
print("\n-----------------------Comparing the results with those obtained during cross-validation for the model with augmented data-------------------")

# Since validation accuracy may be higher than the test accuracy and vice versa, I decided that it would be best to prepare for all cases
if history.history['val_accuracy'][-1] > np.mean(avg_validation_accuracy0):
    print(f"""The hold out test set accuracy, which is equal to {history.history['val_accuracy'][-1]} is better than the validation accuracy, which is equal to {np.mean(avg_validation_accuracy0)}. It is due to the fact that 
    the network was trained on more samples and hence generalized better.""")
else:
    print(f"""The hold out test set accuracy, which is equal to {history.history['val_accuracy'][-1]} is worse than the validation accuracy, which is equal to {np.mean(avg_validation_accuracy0)}. This may be due to a couple of factors:
    1. The validation and test set might come from different distributions
    2. Overfitting - it may be that during training the hyperparameters were overfit to the data""")

if history.history['val_loss'][-1] < np.mean(avg_validation_loss0):
     print(f"""The hold out test set loss, which is equal to {history.history['val_loss'][-1]} is better than the validation loss, which is equal to {np.mean(avg_validation_loss0)}. It is due to the fact that 
    the network was trained on more samples and hence generalized better.""")
else:
    print(f"""The hold out test set loss, which is equal to {history.history['val_loss'][-1]} is worse than the validation loss, which is equal to {np.mean(avg_validation_loss0)}. This may be due to a couple of factors:
    1. The validation and test set might come from different distributions
    2. Overfitting - it may be that during training the hyperparameters were overfit to the data""")


print("Training Second Model on holdout test set\n")
# Building Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# Compiling the model
model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy',tf.keras.metrics.BinaryCrossentropy()])
# Fitting the model
history = model.fit(train_images, train_labels, validation_data = (test_images, test_labels), epochs=10, callbacks = [time_callback])
# save trained model
model.save('saved_model_tf')
print('\nModel saved.\n')
print("\n-----------------------Hold Out Test Set Statistics for model with non-augmented data--------------------------")
model_loss = history.history['val_loss'][-1]
model_acc = history.history['val_accuracy'][-1]
print(f"Loss: {history.history['val_loss'][-1]}, Accuracy: {history.history['val_accuracy'][-1]}, Binary Cross Entropy: {history.history['val_binary_crossentropy'][-1]}\n\n\n\n")
print("\n-----------------------Comparing the results with those obtained during cross-validation for the model with non-augmented data-------------------")
# Since validation accuracy may be higher than the test accuracy and vice versa, I decided that it would be best to prepare for all cases
if history.history['val_accuracy'][-1] > np.mean(avg_validation_accuracy1):
    print(f"""The hold out test set accuracy, which is equal to {history.history['val_accuracy'][-1]} is better than the validation accuracy, which is equal to {np.mean(avg_validation_accuracy1)}. It is due to the fact that 
    the network was trained on more samples and hence generalized better.""")
else:
    print(f"""The hold out test set accuracy, which is equal to {history.history['val_accuracy'][-1]} is worse than the validation accuracy, which is equal to {np.mean(avg_validation_accuracy1)}. This may be due to a couple of factors:
    1. The validation and test set might come from different distributions
    2. Overfitting - it may be that during training the hyperparameters were overfit to the data""")

if history.history['val_loss'][-1] < np.mean(avg_validation_loss1):
     print(f"""The hold out test set loss, which is equal to {history.history['val_loss'][-1]} is better than the validation loss, which is equal to {np.mean(avg_validation_loss1)}. It is due to the fact that 
    the network was trained on more samples and hence generalized better.""")
else:
    print(f"""The hold out test set loss, which is equal to {history.history['val_loss'][-1]} is worse than the validation loss, which is equal to {np.mean(avg_validation_loss1)}. This may be due to a couple of factors:
    1. The validation and test set might come from different distributions
    2. Overfitting - it may be that during training the hyperparameters were overfit to the data""")
##############################################################################################
#                                                                                            #
#       Comparing the two method i.e., the models with and without augmented data            #
#                                                                                            #
##############################################################################################
print("\n-----------------------Comparing the two methods i.e., the models with and without augmented data-------------------")

if cutout_model_acc > model_acc:
  print(f"""The model with the augmented data has accuracy equal to {cutout_model_acc}, which is better than the accuracy of the model with non-augmented data, which is {model_acc}, meaning that the cutout helped the model generalize better.""")
else:
  print(f"""The model with the non-augmented data has accuracy equal to {model_acc}, which is better than the loss of the model with augmented data, which is {cutout_model_loss}, meaning that the cutout didn't help in generalizing better""")

if cutout_model_loss < model_loss:
  print(f"""The model with the augmented data has loss equal to {cutout_model_loss}, which is less than the loss of the model with non-augmented data, which is {model_loss}, meaning that the cutout helped the model generalize better.""")
else:
  print(f"""The model with the non-augmented data has loss equal to {model_loss}, which is less than the loss of the model with augmented data, which is {cutout_model_loss}, meaning that the cutout didn't help in generalizing better""")


