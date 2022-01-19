import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from perceptron import Kernel,Multiclass

def read_dat(path):
    dat = pd.read_csv(path, sep='\s+', header=None).to_numpy()
    return dat

def train_test_split(dataset):
    train_choice = np.random.choice(len(dataset), int(len(dataset) * 0.8), replace=False)
    test_choice = np.random.choice(len(dataset), int(len(dataset) * 0.2), replace=False)
    train = dataset[train_choice,:]
    train_X = train[:, 1:].astype(float)
    train_y = train[:, 0].astype(int)
    test = dataset[test_choice,:]
    test_X = test[:, 1:].astype(float)
    test_y = test[:, 0].astype(int)

    return train_X, train_y, test_X, test_y

def five_fold_cross_validation(train_X, train_y, d, num_class):
    """
    Same code as in 1-2 but with the new "One vs One" method for multiclass classification
    """
    dic = {}
    for dim in d:
        gap = int(len(train_X) / 5)
        kernel = Kernel('poly', dim)
        error_rate = []
        for fold in range(5):
            # get train and test for each fold
            num_sample=len(train_X)
            num_classifier = int(num_class * (num_class - 1) / 2)

            index = np.arange(0, num_sample, 1)
            index_to_test = np.arange(fold*gap,(fold+1)*gap,1)
            index_to_train = list(set(index).difference(set(index_to_test)))
            test_x_fold=train_X[index_to_test]
            test_y_fold=train_y[index_to_test]
            train_x_fold=train_X[index_to_train]
            train_y_fold=train_y[index_to_train]

            # train classifier
            mapped_x = kernel.getFeature(train_x_fold,train_x_fold)
            num_train = len(mapped_x)
            initial_alpha = np.zeros((num_classifier, num_train))
            MP = Multiclass(initial_alpha, mapped_x, train_y_fold, num_class, epoch = 10)
            
            alpha_list = MP.OVOfit() # Training the classifiers

            # Testing accuracy -- mean and std
            test_sample = len(test_x_fold)
            mapped_test = kernel.getFeature(train_x_fold, test_x_fold)
            predict_test = MP.predict(mapped_test, OVO = True)
            test_error = len(np.where((test_y_fold - predict_test) != 0)[0])
            test_error_ratio = test_error / test_sample
            error_rate.append(test_error_ratio)

        dic[f'{dim}']=np.mean(error_rate)
    dim_sort= sorted(dic.items(), key=lambda x: x[1], reverse=False)
    return dim_sort

if __name__ == '__main__':
    file_path = Path('./zipcombo.dat')
    dataset = read_dat(file_path)
    num_class = 10
    d = np.arange(1,8,1)

    testing_error=[]
    best_ds=[]
    test_error_data=[]
    for i in tqdm(range(20)):
        
        train_X, train_y, test_X, test_y = train_test_split(dataset)
        dim_sort=five_fold_cross_validation(train_X,train_y,d,num_class)
        best_d=int(dim_sort[0][0])
        num_classifier = int(num_class * (num_class - 1) / 2)

        # retrain on full 80% training data
        best_kernel=Kernel('poly',best_d)
        mapped_X = best_kernel.getFeature(train_X, train_X)
        num_sample = len(mapped_X)
        initial_alpha = np.zeros((num_classifier, num_sample))
        MP = Multiclass(initial_alpha, mapped_X, train_y, num_class, epoch=10)

        alpha_list = MP.OVOfit()

        # get test errors on he remaining 20% as test data
        # Testing accuracy -- mean and std
        test_sample = len(test_X)
        mapped_test = best_kernel.getFeature(train_X, test_X)
        predict_test = MP.predict(mapped_test,OVO=True)
        test_error = len(np.where((test_y - predict_test) != 0)[0])
        testing_error.append(test_error)

    print(f'Test error: {np.mean(testing_error)}+-{np.std(testing_error)}')
    print(f'Best dimension: {np.mean(best_ds)}+-{np.std(best_ds)}')

    df = pd.DataFrame(index = best_ds)
    df['Test error'] = testing_error
    print(df)
    df.to_csv('Q6_cross_validation.csv')