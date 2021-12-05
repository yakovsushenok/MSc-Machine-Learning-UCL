import pandas as pd
import numpy as np
import SL

# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's

def train_and_test(dataset):
    '''
    This function splits the data into a training and testing set randomly.
    Args:
            dataset - The dataset that we want to split.
    Output:
            returns a splited training set which is 2/3'rds of the data and the remaining is the test set.
    '''
    train_choice=np.random.choice(len(dataset),int(len(dataset)*2/3),replace=False) # Here we are specifying the indices of the training set by a random vector.
    test_choice=np.random.choice(len(dataset),int(len(dataset)/3),replace=False) # We do the same for test set.
    
    train=dataset.loc[train_choice] # Here we are specifying which indexed rows from the original dataset the training set is going to inherit.
    test=dataset.loc[test_choice]  # We do the same for test set.

    train_x=train.iloc[:,:-1] # Here we are specifying the columns which will serve as our predictors for the training set target.
    test_x = test.iloc[:, :-1] # We do the same for test set.

    train_y=train.iloc[:,-1] # Here we are specifying the target of our training set.
    test_y = test.iloc[:, -1]  # We do the same for test set.

    return train_x,train_y,test_x,test_y # Returns the training X and y as well as the test X and y.

def avg_mse(iter,dataset,column):
    """
    This function calculates the average MSE of the specific column by iterating a certain amount of times and averaging the result.
    Args:
            iter - The amount of times we are going to calculate the MSE.
            dataset - The dataset we are dealing with.
            column - The specific predictor we are going to calculate the MSE of.
    Output:
            - Average MSE of the training set
            - Average MSE of the test set
            - Average intercept value for the specific predictor. In particular, we have the following equation when regressing on one predictor y = b_0 + b_1*predictor. The intercept is b_0.
            - Average coefficient of the predictor, b_1.
    """

    train=[] # Initializing our array for the MSE values of each iteration for the training set.
    test=[] # We do the same for test set.
    beta0=[] # Initialzing our arrary for the intercept values of each iteration.
    beta1=[] # Initialziing our array for the coefficient values of each iteration.
    # We will now iterate `iter` times and each iteration we are going to calculate the MSE of the train and test set, the intercept and the coefficient for the relevant predictor. 
    for i in range(iter):
        attr_weight[column] = {} 
        train_x, train_y, test_x, test_y = train_and_test(dataset)

        train_y = np.asarray(train_y)
        test_y = np.asarray(test_y)

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

        train.append(mse_train_)
        test.append(mse_test_)
        beta0.append(w[0])
        beta1.append(w[1])

    return np.mean(train),np.mean(test),np.mean(beta0),np.mean(beta1)

if __name__ == '__main__':
    url = "http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv"
    dataset = pd.read_csv(url)

    #############################################################################################################
    #                                                                                                           #
    #                                     Implementing Naive Regression                                         #
    #                                                                                                           #
    #############################################################################################################

    iternum=20
    naive_mse_train=[]
    naive_mse_test=[]
    for iter in range(iternum):
        
        train_x,train_y,test_x,test_y=train_and_test(dataset)

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)

        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        train_ones=np.ones(train_x.shape[0]).reshape(-1,1)
        test_ones=np.ones(test_x.shape[0]).reshape(-1,1)

        reg=SL.Regression(map=False)

        w=reg.getWeight(train_ones,train_y)
        mse_train=reg.MSE(train_ones.T,train_y)
        mse_test=reg.MSE(test_ones.T,test_y)

        naive_mse_test.append(mse_test)
        naive_mse_train.append(mse_train)

    print(f"Average MSE for 20 iterations: training-{np.mean(naive_mse_train)},"
          f"testing-{np.mean(naive_mse_test)}")

   
    #############################################################################################################
    #                                                                                                           #
    #                                 Implementing Regression with Single predictors                            #
    #                                                                                                           #
    #############################################################################################################
    
    attr_weight = {}
    for column in dataset:
        if column == 'MEDV':
            break
        attr_weight[column] = {}
        mse_train_,mse_test_,beta0,beta1=avg_mse(20,dataset,column)

        attr_weight[column]['beta_0'] = beta0
        attr_weight[column]['beta_1'] = beta1

        attr_weight[column]['mse_train'] = mse_train_
        attr_weight[column]['mse_test'] = mse_test_
    
    print(attr_weight)
    df = pd.DataFrame(attr_weight)
    print(df)

    #############################################################################################################
    #                                                                                                           #
    #                                 Implementing Regression with All Predictors                               #
    #                                                                                                           #
    #############################################################################################################

    train_attr_mse=[]
    test_attr_mse=[]
    for i in range(iternum):
        train_x, train_y, test_x, test_y = train_and_test(dataset)

        reg=SL.Regression()
        w=reg.getWeight(train_x,train_y)

        mse_train=reg.MSE(train_x.T,train_y)
        mse_test=reg.MSE(test_x.T,test_y)

        train_attr_mse.append(mse_train)
        test_attr_mse.append(mse_test)

    print(f'MSE for the training sample is: {np.mean(train_attr_mse)}, MSE for the testing sample is: {np.mean(test_attr_mse)}')



