import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from perceptron import Kernel, Multiclass

def read_dat(path):
    '''
    This function reads in the file used for the zipcombo.dat data file
    Args:
            path - file path
    Returns
            data - pandas dataset
    '''
    dat = pd.read_csv(path, sep='\s+', header=None).to_numpy()
    return dat

def train_test_split(dataset):
    '''
    This function splits the whole dataset as 80% train and 20% test
    Args:
            dataset - dataset to be split
    Returns:
            train_x, train_y, test_x, test_y - Training/testing samples and labels 
    '''
    train_choice = np.random.choice(len(dataset), int(len(dataset) * 0.8), replace=False)
    test_choice = np.random.choice(len(dataset), int(len(dataset) * 0.2), replace=False)
    train = dataset[train_choice,:]
    train_X = train[:, 1:].astype(float)
    train_y = train[:, 0].astype(int)
    test = dataset[test_choice,:]
    test_X = test[:, 1:].astype(float)
    test_y = test[:, 0].astype(int)

    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    
    # preparation
    file_path = Path('./zipcombo.dat')
    dataset = read_dat(file_path)
    num_class=10

    # Initializing the error arrays.
    training_error_mean = []
    testing_error_mean = []
    training_error_std = []
    testing_error_std = []
    
    # Initializing the array for the dimensions of the polynomial as well as the pandas dataframe for later usage
    dimension = np.arange(1, 8, 1)
    df = pd.DataFrame(index = dimension)
    
    # Starting the loop
    for d in tqdm(range(1, 8)):
        
        # Initializing the error array for the specific d
        training_error = []
        testing_error = []
        
        # Initializing the polynomial kernel
        kernel=Kernel('poly', d)
        
        # Iterating through 20 runs to get the average errors per d
        for i in range(20):
            train_X, train_y, test_X, test_y = train_test_split(dataset) # Splitting the dataset
            mapped_X = kernel.getFeature(train_X, train_X) # Getting the kernel matrix
            num_sample = len(mapped_X)
            
            # initializing the alpha classifiers which a zero matrix with size (number of class)*(num of sample)
            initial_alpha = np.zeros((num_class,num_sample))
            MP = Multiclass(initial_alpha, mapped_X, train_y, num_class, epoch=10)

            alpha_list = MP.fit() # Training the classifiers

            # Training accuracy -- mean and standard deviation
            predict = MP.predict(mapped_X)
            train_error = len(np.where((train_y-predict)!=0)[0])
            train_error_ratio = train_error/num_sample
            training_error.append(train_error_ratio)

            # Testing accuracy -- mean and standard deviation
            test_sample = len(test_X)
            mapped_test = kernel.getFeature(train_X,test_X)
            predict_test = MP.predict(mapped_test)
            test_error = len(np.where((test_y-predict_test)!=0)[0])
            print(f'run {i + 1}: {test_error} points incorrectly predicted out of a total of {test_sample} points')
            test_error_ratio = test_error/test_sample
            testing_error.append(test_error_ratio)

        # Appending the mean errors 
        training_error_mean.append(np.mean(training_error))
        testing_error_mean.append(np.mean(testing_error))
        training_error_std.append(np.std(training_error))
        testing_error_std.append(np.std(training_error))

    df['Train_mean_e'] = training_error_mean
    df['Train_std_e'] = training_error_std
    df['test_mean_e'] = testing_error_mean
    df['test_std_e'] = testing_error_std
    print(df)
    # Plotting the mean errors against their corresponding d
    plt.figure(figsize = (12,6))
    plt.plot(dimension, training_error_mean, '#F08080', label='Training Error')
    plt.plot(dimension, testing_error_mean, '#2E8B57', label='Testing Error')
    plt.legend()
    plt.xlabel('Dimensions')
    plt.ylabel('Error %')
    plt.title('Error of training and testing data in different dimensions')
    plt.show()
    plt.savefig('Q1_train_test_error.png')