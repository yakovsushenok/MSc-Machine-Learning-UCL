import numpy as np
import pandas as pd
import copy
import SL
from tqdm import tqdm
import seaborn as sbn
import matplotlib.pyplot as plt



def train_and_test(dataset):
    '''
    This function splits the data into a training and testing set randomly.
    Args:
            dataset - The dataset that we want to split.
    Output:
            returns a splited training set which is 2/3'rds of the data and the remaining is the test set.
    '''
    train_choice=np.random.choice(len(dataset),340,replace=False) # Here we are specifying the indices of the training set by a random vector.
    test_choice=np.random.choice(len(dataset),120,replace=False)  # We do the same for test set.

    train=dataset.loc[train_choice] # Here we are specifying which indexed rows from the original dataset the training set is going to inherit.
    test=dataset.loc[test_choice] # We do the same for test set.

    train_x=train.iloc[:,:-1] # # Here we are specifying the columns which will serve as our predictors for the training set target.
    train_y=train.iloc[:,-1] # We do the same for test set.

    test_x = test.iloc[:, :-1]  # Here we are specifying the target of our training set.
    test_y = test.iloc[:, -1] # We do the same for test set.

    return train_x,train_y,test_x,test_y # Returns the training X and y as well as the test X and y.

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

def five_fold_cross_validation(train_x, train_y, test_x, test_y,gamma_power,sigma_power):
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
                test_y_fold = train_y.iloc[fold * gap:(fold + 1) * gap].values

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

                # calculate MSE
                krr = SL.KRR()
                krr.regress(gamma, kernel_matrix, train_y_fold)

                kernel_test = np.zeros((train_x_fold.shape[0], test_x_fold.shape[0]))
                for a in range(train_x_fold.shape[0]):
                    for b in range(test_x_fold.shape[0]):
                        kernel_test[a][b] = gaussian_kernel(sigma, train_x_fold[a, :], test_x_fold[b, :])

                predict = krr.predict(kernel_test)
                mse = krr.MSE(predict.T, test_y_fold)
                mses.append(mse)

            mse_matrix[i][j] = np.mean(mses)
            mse_dic[f'gamma=2^{gamma_power[j]},sigma=2^{sigma_power[i]}'] = np.mean(mses)

    mse_dic = sorted(mse_dic.items(), key=lambda x: x[1], reverse=False)

    return mse_dic,mse_matrix


def train_and_test_ridge_regression_task_a(dataset,best_sigma,best_gamma):
    '''
    This function calculates the mse for kernel ridge regression both for the train and test set for the best sigma and gamma pair.
    Args:
            dataset - The boston housing dataset.
            best_sigma - The sigma in the (gamma,sigma) pair which gives the best MSE.
            best_gamma - The gamma in the (gamma,sigma) pair which gives the best MSE.
    Output:
            Returns the MSE for the best (sigma,gamma) pair for both train and test sets.
    '''
    train_mses = []
    test_mses = []
    train_x, train_y, test_x, test_y = train_and_test(dataset)

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    kernel_matrix = np.zeros((train_x.shape[0], train_x.shape[0]))
    for a in range(train_x.shape[0]):
        for b in range(train_x.shape[0]):
            kernel_matrix[a][b] = gaussian_kernel(2 ** best_sigma, train_x[a, :], train_x[b, :])

    krr = SL.KRR()
    krr.regress(2 ** int(best_gamma), kernel_matrix, train_y)
    predict_train = krr.predict(kernel_matrix)
    mse_train = krr.MSE(predict_train, train_y)
    train_mses.append(mse_train)

    kernel_test = np.zeros((train_x.shape[0], test_x.shape[0]))
    for a in range(train_x.shape[0]):
        for b in range(test_x.shape[0]):
            kernel_test[a][b] = gaussian_kernel(2 ** best_sigma, train_x[a, :], test_x[b, :])

    predict_test = krr.predict(kernel_test)
    mse_test = krr.MSE(predict_test.T, test_y)
    test_mses.append(mse_test)

    return train_mses,test_mses

if __name__ == '__main__':
    url = "http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv"
    dataset = pd.read_csv(url)
    gamma_power=np.arange(-40,-25,1)
    sigma_power=np.arange(7,13,0.5)

    train_x, train_y, test_x, test_y = train_and_test(dataset)

    #############################################################################################################
    #                                                                                                           #
    #                            5(a) calculating the best (sigma,gamma) pair                                   #
    #                                                                                                           #
    #############################################################################################################

    mse_dic,mse_matrix=five_fold_cross_validation(train_x,train_y,test_x,test_y,gamma_power,sigma_power)
    print(mse_dic)

    #############################################################################################################
    #                                                                                                           #
    #                          5(c) Calculating the MSE for best (sigma,gamma) pair                             #
    #                                                                                                           #
    #############################################################################################################

    train_mse, test_mse = train_and_test_ridge_regression_task_a(dataset,9,-27)
    print(train_mse,test_mse)

    #############################################################################################################
    #                                                                                                           #
    #               5(b) Plotting the heat map for sigmma gamma values and their corresponding MSE              #
    #                                                                                                           #
    #############################################################################################################
    plt.figure(figsize=[10, 10])
    sbn.heatmap(mse_matrix, vmax=100, annot=True, linewidth=0.2, linecolor="orange")
    plt.ylabel("Standard deviation parameter of Gaussian kernel (power of 2)")
    plt.xlabel("Regularisation parameter of Ridge Regression (power of 2)")
    plt.yticks(np.arange(0, 13, 1), np.arange(7, 13.5, 0.5))
    plt.xticks(np.arange(0, 15, 1), np.arange(-40, -25, 1))
    plt.show()

    print(f'The minimun MSE is: {mse_matrix.min()},'
          f' where standard deviation parameter ofr gaussian kernel is 2^{sigma_power[np.where(mse_matrix==mse_matrix.min())[0][0]]}, '
          f'and regularisation parameter of ridge regression is 2^{gamma_power[np.where(mse_matrix==mse_matrix.min())[1][0]]}')





