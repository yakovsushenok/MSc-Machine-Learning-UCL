import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from perceptron import Kernel,Multiclass
import seaborn as sns

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
    train = dataset[train_choice, :]
    train_X = train[:, 1:].astype(float)
    train_y = train[:, 0].astype(int)
    test = dataset[test_choice, :]
    test_X = test[:, 1:].astype(float)
    test_y = test[:, 0].astype(int)

    return train_X, train_y, test_X, test_y

def five_fold_cross_validation(train_X, train_y, d, num_class):
    """
    This function performs 5-fold cross validation for all of the value of d and returns the d* with the smallest cross validation error.
    Args:
            train_X - Training samples
            train_y - Training labels
            d - list of d candidates
            num_class - number of classes in the dataset
    Returns:
            d* - optimal d
    """
    dic = {}
    for dim in d: # looping through all d's
        
        gap = int(len(train_X) / 5) 
        kernel = Kernel('poly', dim)
        error_rate = []
        
        for fold in range(5): # Looping through the 5 train-validation pairs
            
            # Setting up the training and validation set
            num_sample = len(train_X)
            index = np.arange(0, num_sample,1)
            index_to_test = np.arange(fold * gap, (fold + 1) * gap, 1)
            index_to_train = list(set(index).difference(set(index_to_test)))
            test_x_fold = train_X[index_to_test]
            test_y_fold = train_y[index_to_test]
            train_x_fold = train_X[index_to_train]
            train_y_fold = train_y[index_to_train]

            # training the classifier
            mapped_x = kernel.getFeature(train_x_fold, train_x_fold)
            num_train = len(mapped_x)
            initial_alpha = np.zeros((num_class, num_train))
            MP = Multiclass(initial_alpha, mapped_x, train_y_fold, num_class, epoch=10)
            alpha_list = MP.fit()

            # Testing accuracy -- mean and standard deviation
            test_sample = len(test_x_fold)
            mapped_test = kernel.getFeature(train_x_fold, test_x_fold)
            predict_test = MP.predict(mapped_test)
            test_error = len(np.where((test_y_fold - predict_test) != 0)[0])
            test_error_ratio = test_error / test_sample
            error_rate.append(test_error_ratio)
        
        # Appending the results to the dictionary
        dic[f'{dim}'] = np.mean(error_rate)
    
    # Sorting the dictionary to get the optimal value of d
    dim_sort = sorted(dic.items(), key=lambda x: x[1], reverse=False)
    
    return dim_sort




if __name__ == '__main__':
    
    # Preparing the data  
    file_path = Path('./zipcombo.dat')
    dataset = read_dat(file_path)
    num_class = 10
    d=np.arange(1,8,1)

    # Initializing the lists for the testing errors, the confusion matrix and the "hard to predict numbers".
    testing_error = []
    best_ds = []
    confusion_matrix = []
    prediction_hard = np.zeros((num_class,1))
    test_error_data=[]
    
    for i in tqdm(range(20)):
        
        # Splitting the dataset 
        train_X, train_y, test_X, test_y = train_test_split(dataset)
        dim_sort = five_fold_cross_validation(train_X,train_y,d,num_class)
        best_d = int(dim_sort[0][0])

        # retrain on full 80% training data
        best_kernel=Kernel('poly',best_d)
        mapped_X = best_kernel.getFeature(train_X, train_X)
        num_sample = len(mapped_X)
        initial_alpha = np.zeros((num_class, num_sample))
        MP = Multiclass(initial_alpha, mapped_X, train_y, num_class, epoch=10)

        # Training the classifiers
        alpha_list = MP.fit()

        # get test errors on he remaining 20% as test data
        # Testing accuracy -- mean and std
        test_sample = len(test_X)
        mapped_test = best_kernel.getFeature(train_X, test_X)
        predict_test = MP.predict(mapped_test)
        test_error = len(np.where((test_y - predict_test) != 0)[0])
        
        # preparation for draw hard digits
        for where_error in np.where((test_y-predict_test) != 0)[0]:
            test_error_data.append((test_X[where_error], test_y[where_error]))

        test_error_ratio = test_error / test_sample
        testing_error.append(test_error_ratio)
        best_ds.append(best_d)

        # confusion matrix for each run
        confusion_single = np.zeros((10, 10))
        label_num = np.zeros((10, 1))
        for i, j in zip(test_y, predict_test):
            label_num[i] += 1
            if i != j:
                confusion_single[i, j] += 1

                # (d) find 5 hard predicted number
                prediction_hard[i] += 1

                line=1

        confusion_single = confusion_single/label_num
        confusion_matrix.append(confusion_single)


    # ---------------------------------------------------------------------#
    # ------------------------------- (Q2) --------------------------------#
    # ---------------------------------------------------------------------#
    # Getting the mean errors and standard deviation

    print(f'Test error: {np.mean(testing_error)}+-{np.std(testing_error)}')
    print(f'Best dimension: {np.mean(best_ds)}+-{np.std(best_ds)}')

    df = pd.DataFrame(index = best_ds)
    df['Test error'] = testing_error
    print(df)
    df.to_csv('table_Q2.csv')
    


    # ---------------------------------------------------------------------#
    # ------------------------------- (Q3) --------------------------------#
    # ---------------------------------------------------------------------#
    # Confusion matrix

    mean_confusion_pair = np.mean(confusion_matrix, axis = 0)
    std_confusion_pair = np.std(confusion_matrix, axis = 0)
    df_confusion=pd.DataFrame(columns = np.arange(1,11,1), index = np.arange(1, 11, 1))
    # store result in dataframe and save to csv
    for i in range(10): 
        for j in range(10):
            df_confusion.iloc[i, j]=f'{np.round(mean_confusion_pair[i, j], 4)}+-{np.round(std_confusion_pair[i, j], 4)}'

    df_confusion.to_csv('table_Q3.csv')
    
    try:
    
        plt.figure(figsize = (12, 6))
        sns.heatmap(mean_confusion_pair, linewidth = 0.2, linecolor = "orange")
        plt.show()
    
    except:
        print('Fail to draw heatmap of confusion matrix')



    # ---------------------------------------------------------------------#
    # ------------------------------- (Q4) ---------------------------------#
    # ---------------------------------------------------------------------#
    # Printing out the hardest 5 numbers to predict

    prediction_hard=list(int(x[0]) for x in prediction_hard)
    worst_5_number = np.argsort(prediction_hard)[:5]

    worst_5_data=np.zeros((5,dataset.shape[1]-1))
    for n in range(len(worst_5_data)):
        
        line=1
        for wrong_pre in test_error_data:
        
            if wrong_pre[1]==worst_5_number[n] and line==1:
                worst_5_data[n,:]=wrong_pre[0]
        
            else:continue
            line+=1

    plot_num=1
    for i in range(len(worst_5_number)):
        
        plt.subplot(1,5,plot_num)
        img = np.reshape(worst_5_data[i,:], (16, 16))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{worst_5_number[i]}')
        plot_num += 1
    
    plt.show()
    plt.savefig('Q4_hard_digit.png')



