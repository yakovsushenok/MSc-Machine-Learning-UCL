import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from perceptron import Kernel, Multiclass
import time

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
    test = dataset[test_choice, :]
    test_X = test[:, 1:].astype(float)
    test_y = test[:, 0].astype(int)

    return train_X,train_y,test_X,test_y

def select_best_S(initial_S, dataset):
    """
    This function serves as assistance for the initial experiments we did to find a suitable range of c values to cross validate over.
    """
    error_test_S = []
    for c in initial_S:
        kernel = Kernel('gaussian', c)
        testing_error=[]
        for run in tqdm(range(20)):
            
            train_X, train_y, test_X, test_y = train_test_split(dataset)
            mapped_X = kernel.getFeature(train_X, train_X)
            num_sample = len(mapped_X)
            
            # initial alpha is a zero matrix of size (number of class)*(num of sample)
            initial_alpha = np.zeros((num_class, num_sample))
            MP = Multiclass(initial_alpha, mapped_X, train_y, num_class, epoch=10)
            alpha_list = MP.fit()

            # Testing accuracy -- mean and std
            test_sample = len(test_X)
            mapped_test = kernel.getFeature(train_X, test_X)
            predict_test = MP.predict(mapped_test)
            test_error = len(np.where((test_y - predict_test) != 0)[0])
            test_error_ratio = test_error / test_sample
            testing_error.append(test_error_ratio)
        
        error_test_S.append(np.mean(testing_error))
    
    suitable_range=list(initial_S[i] for i in np.argsort(error_test_S))
    print(np.sort(error_test_S))
    print(suitable_range)

def basicResult(S,dataset,num_class):
    """
    This code is just a slight modification of what whas done in Q1-2, but considering the parameter c now
    """
    training_error_mean = []
    testing_error_mean = []
    training_error_std = []
    testing_error_std = []
    df = pd.DataFrame(index=S)
    for c in tqdm(range(len(S))):
        training_error = []
        testing_error = []

        kernel=Kernel('gaussian',S[c])
        print(f'c={S[c]}')

        for i in range(20):
            train_X, train_y, test_X, test_y = train_test_split(dataset)
            mapped_X = kernel.getFeature(train_X, train_X)
            num_sample = len(mapped_X)
            # initial alpha is a zero matrix which size is (number of class)*(num of sample)
            initial_alpha=np.zeros((num_class, num_sample))
            MP = Multiclass(initial_alpha, mapped_X, train_y, num_class, epoch = 10)

            alpha_list=MP.fit() # get the trained alpha matrix

            # Training accuracy -- mean and ste
            predict = MP.predict(mapped_X)
            train_error = len(np.where((train_y-predict)!=0)[0])
            train_error_ratio=train_error/num_sample
            training_error.append(train_error_ratio)

            # Testing accuracy -- mean and std
            test_sample = len(test_X)
            mapped_test = kernel.getFeature(train_X, test_X)
            predict_test = MP.predict(mapped_test)
            test_error = len(np.where((test_y-predict_test)!=0)[0])
            print(f'run {i+1}: error predict {test_error} points with total {test_sample} points')
            test_error_ratio = test_error/test_sample
            testing_error.append(test_error_ratio)

        training_error_mean.append(np.mean(training_error))
        testing_error_mean.append(np.mean(testing_error))
        training_error_std.append(np.std(training_error))
        testing_error_std.append(np.std(training_error))

    df['Train_mean_e'] = training_error_mean
    df['Train_std_e'] = training_error_std
    df['test_mean_e'] = testing_error_mean
    df['test_std_e'] = testing_error_std
    print(df)

    plt.figure(figsize=(12, 6))
    plt.plot(S, training_error_mean, '#F08080', label='Training Accuracy')
    plt.plot(S, testing_error_mean, '#2E8B57', label='Testing Accuracy')
    plt.legend()
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy of training and testing data in different dimensions (Gaussian)')
    plt.show()

    return df

def five_fold_cross_validation(train_X,train_y,S,num_class):
    """
    This is the same code as in Q2 but with c
    """
    dic={}
    for c in S:
        gap = int(len(train_X) / 5)
        kernel = Kernel('gaussian', c)
        error_rate=[]
        for fold in range(5):
           
            # get train and test for each fold
            num_sample = len(train_X)
            index=np.arange(0, num_sample, 1) 
            index_to_test = np.arange(fold * gap,(fold + 1) * gap, 1)
            index_to_train = list(set(index).difference(set(index_to_test)))
            test_x_fold=train_X[index_to_test]
            test_y_fold=train_y[index_to_test]
            train_x_fold=train_X[index_to_train]
            train_y_fold=train_y[index_to_train]

            # train classifier
            mapped_x = kernel.getFeature(train_x_fold, train_x_fold)
            num_train = len(mapped_x)
            initial_alpha = np.zeros((num_class, num_train))
            MP = Multiclass(initial_alpha, mapped_x, train_y_fold, num_class, epoch = 10)
            alpha_list = MP.fit()

            # Testing accuracy -- mean and std
            test_sample = len(test_x_fold)
            mapped_test = kernel.getFeature(train_x_fold, test_x_fold)
            predict_test = MP.predict(mapped_test)
            test_error = len(np.where((test_y_fold - predict_test) != 0)[0])
            test_error_ratio = test_error / test_sample
            error_rate.append(test_error_ratio)

        dic[f'{c}'] = np.mean(error_rate)
    
    dim_sort= sorted(dic.items(), key=lambda x: x[1], reverse=False)
    return dim_sort

def cross_validation(S,dataset):
    """
    This function does 20 runs with the best c which are obtained by 5-fold cross-validation
    """
    testing_error = []
    best_ds = []
    
    for i in tqdm(range(20)):
        
        train_X, train_y, test_X, test_y = train_test_split(dataset)
        dim_sort = five_fold_cross_validation(train_X,train_y,S,num_class)
        best_d = np.round(float(dim_sort[0][0]),2)

        # retrain on full 80% training data
        best_kernel = Kernel('gaussian',best_d)
        mapped_X = best_kernel.getFeature(train_X, train_X)
        num_sample = len(mapped_X)
        initial_alpha = np.zeros((num_class, num_sample))
        MP = Multiclass(initial_alpha, mapped_X, train_y, num_class, epoch=10)

        alpha_list = MP.fit()

        # get test errors on he remaining 20% as test data
        # Testing accuracy -- mean and std
        test_sample = len(test_X)
        mapped_test = best_kernel.getFeature(train_X, test_X)
        predict_test = MP.predict(mapped_test)
        test_error = len(np.where((test_y - predict_test) != 0)[0])
        test_error_ratio = test_error / test_sample
        print(f'run {i + 1}: error predict {test_error} points with total {test_sample} points, best width is {best_d}')
        testing_error.append(test_error_ratio)
        best_ds.append(best_d)
    return testing_error,best_ds

if __name__ == '__main__':
    # preparation
    file_path = Path('./zipcombo.dat')
    dataset = read_dat(file_path)
    num_class=10
    """
     -----------------------------------------------------------------------------------------
                                 initial experiment on set S
     -----------------------------------------------------------------------------------------
     S=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
     start=time.time()
     select_best_S(S,dataset)
     end=time.time()
     print(f'use {end-start}s to find reasonable S')


    -----------------------------------------------------------------------------------------
     result (20 runs)
     test error: [0.00640129 0.00734266 0.00879505 0.02369554 0.09518558 0.11226466 0.14725659]
     related S: [0.05, 0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
     then we decide to set S between 0.01 and 0.1
    -----------------------------------------------------------------------------------------
    """
    
    S = np.arange(0.01,0.11,0.01)
    
    
    # -----------------------basic result--------------------------
    
    df_basicResult=basicResult(S,dataset,num_class)
    df_basicResult.to_csv('Q4_basicResult.csv')

    
    # ---------------------cross validation------------------------
    
    testing_error,best_ds=cross_validation(S,dataset)
    print(f'Test error: {np.mean(testing_error)}+-{np.std(testing_error)}')
    print(f'Best dimension: {np.mean(best_ds)}+-{np.std(best_ds)}')

    df_cross = pd.DataFrame(index=best_ds)
    df_cross['Test error'] = testing_error
    print(df_cross)
    df_cross.to_csv('Q5_cross_validation.csv')


