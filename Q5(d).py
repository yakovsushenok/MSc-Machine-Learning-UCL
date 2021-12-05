import numpy as np
import pandas as pd
import copy
import SL
from tqdm import tqdm
import seaborn as sbn
import matplotlib.pyplot as plt
import time

def train_and_test(dataset):
    '''
    This function splits the data into a training and testing set randomly.
    Args:
            dataset - The dataset that we want to split.
    Output:
            returns a splited training set which is 2/3'rds of the data and the remaining is the test set.
    '''
    train_choice=np.random.choice(len(dataset),int(len(dataset)*2/3),replace=False)
    test_choice=np.random.choice(len(dataset),int(len(dataset)/3),replace=False)
    train=dataset.loc[train_choice]

    test=dataset.loc[test_choice]
    train_x=train.iloc[:,:-1] # columns except MEDV
    train_y=train.iloc[:,-1] # column MEDV as y
    test_x = test.iloc[:, :-1]  # columns except MEDV
    test_y = test.iloc[:, -1]  # column MEDV as y

    return train_x,train_y,test_x,test_y

def gaussian_kernel(sigma,x1,x2):
    '''
    This function calculate the value of the Gaussian kernel.
    Args:
            sigma - The sigma which is used in the Gaussian kernel.
            x1 - Vector x_i.
            x2 - Vector x_j.
    Output:
            Returns the value of the gaussian kernel K(x_i,x_j)
    '''
    return np.exp(-(np.linalg.norm(x1-x2)**2)/(2*sigma**2))

def five_fold_cross_validation(gamma_power,sigma_power):
    '''
    This function performs 5 fold cross validation and calculate the MSE for all of the (sigma,gamma) pairs.
    Args:
            train_x - The predictors of the training set.
            train_y - The target variable of the training set.
            test_x - The predictors of the test set.
            test_y - The target variable of the test set.
            gamma_power - A list of the powers for the gamma values. So if the list of gammas is [2^{-40},....,2^{-26}], then the gamma_power list is [-40,...,-26].
            sigma_power - Same as the gamma_power list but for the powers of sigma.
    Output:
            Returns a matrix of the MSE's and a dictionary of MSE's which is sorted by best sigmma.
    :return: mse matrix and a dictionary of best gamma - sigma pairs sorted by best MSE. 
    '''
    train_x, train_y, test_x, test_y = train_and_test(dataset)
    mse_matrix = np.asarray([[0.0] * gamma_power.shape[0] for x in range(sigma_power.shape[0])])
    print(mse_matrix.shape)
    mse_dic = {}

    for i in range(len(sigma_power)):
        for j in range(len(gamma_power)):
            gamma = 2 ** int(gamma_power[j])
            sigma = 2 ** sigma_power[i]
            mses = []
            gap = int(len(train_x) / 5)
            for fold in tqdm(range(5)):
                # get test
                test_x_fold = train_x.iloc[fold * gap:(fold + 1) * gap].values
                test_y_fold  = train_y.iloc[fold * gap:(fold + 1) * gap].values

                # get train
                index = train_x.iloc[fold * gap:(fold + 1) * gap].index.tolist()
                train_x_fold = copy.deepcopy(train_x)  # create an independent train_x in this loop
                train_y_fold = copy.deepcopy(train_y)
                train_x_fold = train_x_fold.drop(index=index)  # drop index which belong to test fold
                train_y_fold = train_y_fold.drop(index=index)
                train_x_fold = train_x_fold.values
                train_y_fold = train_y_fold.values

                kernel_matrix = np.zeros((train_x_fold.shape[0], train_x_fold.shape[0]))

                for a in range(train_x_fold.shape[0]):
                    for b in range(train_x_fold.shape[0]):
                        kernel_matrix[a][b] = gaussian_kernel(sigma, train_x_fold[a, :], train_x_fold[b, :])

                krr = SL.KRR()
                krr.regress(gamma, kernel_matrix, train_y_fold)

                kernel_test = np.zeros((train_x_fold.shape[0], test_x_fold.shape[0]))
                for a in range(train_x_fold.shape[0]):
                    for b in range(test_x_fold.shape[0]):
                        kernel_test[a][b] = gaussian_kernel(sigma, train_x_fold[a, :], test_x_fold[b, :])

                predict = krr.predict(kernel_test)
                mse_test = krr.MSE(predict.T, test_y_fold)
                mses.append(mse_test)

            mse_matrix[i][j] = np.mean(mses)
            mse_dic[f'gamma={gamma},sigma={sigma}'] = np.mean(mses)

    mse_dic = sorted(mse_dic.items(), key=lambda x: x[1], reverse=False)

    return mse_dic,mse_matrix

if __name__ == '__main__':
    url = "http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv"
    dataset = pd.read_csv(url)

    # define result table
    df_result=pd.DataFrame(columns=['MSE train','MSE test'],dtype=str)

    #############################################################################################################
    #                                                                                                           #
    #                                     Implementing Naive Regression                                         #
    #                                                                                                           #
    #############################################################################################################
    iternum = 50
    naive_mse_train = []
    naive_mse_test = []
    for iter in range(iternum):
        train_x, train_y, test_x, test_y = train_and_test(dataset)
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        train_ones = np.ones(train_x.shape[0]).reshape(-1, 1)
        test_ones = np.ones(test_x.shape[0]).reshape(-1, 1)
        reg = SL.Regression(map=False)
        w = reg.getWeight(train_ones, train_y)
        mse_train = reg.MSE(train_ones.T, train_y)
        mse_test = reg.MSE(test_ones.T, test_y)

        naive_mse_test.append(mse_test)
        naive_mse_train.append(mse_train)

    print(f"Average MSE for 20 iterations: training-{np.mean(naive_mse_train)},"
          f"testing-{np.mean(naive_mse_test)}")
    std_train=np.std(naive_mse_train)
    std_test=np.std(naive_mse_test)

    df_result.loc['Naive Regression']=[f'{round(np.mean(naive_mse_train),4)}+-{round(std_train,4)}',
                                       f'{round(np.mean(naive_mse_test),4)}+-{round(std_test,4)}']

    #############################################################################################################
    #                                                                                                           #
    #                                 Implementing Regression with Single predictors                            #
    #                                                                                                           #
    #############################################################################################################

    for column in dataset:
        if column == 'MEDV':
            break

        mse_train_d=[]
        mse_test_d=[]
        for iter in range(iternum):
            train_x, train_y, test_x, test_y = train_and_test(dataset)
            train_x_attr = train_x[column].values
            test_x_attr = test_x[column].values

            matrix_train = np.zeros((len(train_x), 2))
            matrix_test = np.zeros((len(test_x), 2))

            for i in range(len(train_x)):
                matrix_train[i][0] = train_x_attr[i]
                matrix_train[i][1] = 1

            for j in range(len(test_x)):
                matrix_test[j][0] = test_x_attr[j]
                matrix_test[j][1] = 1

            reg = SL.Regression()
            w = reg.getWeight(matrix_train, train_y)
            mse_train_ = reg.MSE(matrix_train.T, train_y)
            mse_test_ = reg.MSE(matrix_test.T, test_y)

            mse_train_d.append(mse_train_)
            mse_test_d.append(mse_test_)

        mean_train=np.mean(mse_train_d)
        std_train=np.std(mse_train_d)
        mean_test=np.mean(mse_test_d)
        std_test=np.std(mse_test_d)

        df_result.loc[column]=[f'{round(mean_train,4)}+-{round(std_train,4)}',
                               f'{round(mean_test,4)}+-{round(std_test,4)}']


    #############################################################################################################
    #                                                                                                           #
    #                                 Implementing Kernel Ridge Regression                                      #
    #                                                                                                           #
    #############################################################################################################

    start = time.perf_counter()

    gamma_power = np.arange(-40, -25, 1)
    sigma_power = np.arange(7, 13, 0.5)
    train_mses = []
    test_mses = []
    iternum_k=20
    for iter in range(iternum_k):
        mse_dic, mse_matrix = five_fold_cross_validation(gamma_power, sigma_power)

        best_gamma=gamma_power[np.where(mse_matrix==mse_matrix.min())[1][0]]
        best_sigma=sigma_power[np.where(mse_matrix==mse_matrix.min())[0][0]]

        train_x, train_y, test_x, test_y = train_and_test(dataset)
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        kernel_matrix = np.zeros((train_x.shape[0], train_x.shape[0]))
        for a in range(train_x.shape[0]):
            for b in range(train_x.shape[0]):
                kernel_matrix[a][b] = gaussian_kernel(2**best_sigma, train_x[a, :], train_x[b, :])

        krr = SL.KRR()
        krr.regress(2**int(best_gamma), kernel_matrix, train_y)
        predict_train=krr.predict(kernel_matrix)
        mse_train=krr.MSE(predict_train,train_y)
        train_mses.append(mse_train)

        kernel_test = np.zeros((train_x.shape[0], test_x.shape[0]))
        for a in range(train_x.shape[0]):
            for b in range(test_x.shape[0]):
                kernel_test[a][b] = gaussian_kernel(2**best_sigma, train_x[a, :], test_x[b, :])

        predict_test = krr.predict(kernel_test)
        mse_test = krr.MSE(predict_test.T, test_y)
        test_mses.append(mse_test)

    mean_train=np.mean(train_mses)
    std_train=np.std(train_mses)
    mean_test=np.mean(test_mses)
    std_test=np.mean(test_mses)
    df_result.loc['Kernel Ridge Regression']=[f'{round(mean_train,4)}+-{round(std_train,4)}',
                               f'{round(mean_test,4)}+-{round(std_test,4)}']


    print(df_result)

    end = time.perf_counter()

    print(f'It takes totally {start-end}')

