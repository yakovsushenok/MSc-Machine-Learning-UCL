import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def data_generate(m, n, winnow = False):
    '''
    :param m: number of examples
    :param n: dimension(n) of the data
    :param winnow: if we use winnow algorithm
    :return: generated X data and y data
    '''
    if winnow == False:
        X = np.random.choice([-1,1], (m, n))
        y = X[:,0] # label of a pattern x is just its first coordinate
    else:
        X = np.random.choice([0,1], (m, n))
        y = X[:,0]
    return X, y

def perceptron(train_X,train_y,test_X):
    '''
    weight: n*1 matrix; train_X: m*n matrix
    :param train_X: X data for train
    :param train_y: y data for train
    :param test_X: X data for test
    :return:
    '''
    m,n=train_X.shape
    weights=np.zeros((n))
    for i in range(m):
        # predict
        y_i_hat=np.sign(weights.dot(train_X[i,:]))
        mistake=0
        # update
        if y_i_hat*train_y[i]<=0:
            weights+=train_y[i]*train_X[i,:]
            mistake+=1

    y_pred=np.sign(weights.dot(test_X.T))
    # because sign(0)=0 but we want sign(0)=-1, so we have to change
    # y_pred=[1 if pred > 0 else -1 for pred in y_pred]
    return y_pred

def winnow(train_X,train_y,test_X):
    m,n=train_X.shape
    weight=np.ones((n)) # n: dimension of data, weight: n*1
    for i in range(m):
        if weight.dot(train_X[i,:])>=n:
            y_pred_i=1
        else:
            y_pred_i=0

        if train_y[i]==1 and y_pred_i==0:
            weight=np.multiply(weight,2,where=(train_X[i]>0))
        elif train_y[i]==0 and y_pred_i==1:
            weight=np.divide(weight,2,where=(train_X[i]>0))

    y_pred=np.where(weight.dot(test_X.T) < n, 0, 1)
    return y_pred

def least_square(train_X,train_y,test_X):
    # consider pseudoinverse case
    w=np.linalg.pinv(train_X).dot(train_y)
    y_pred=np.sign(test_X.dot(w))
    # because sign(0)=0 but we want sign(0)=-1, so we have to change
    # y_pred = [1 if pred > 0 else -1 for pred in y_pred]
    return y_pred

def one_NN(train_X,train_y,test_X):
    '''
    :param train_X: X dataset m*n
    :param train_y: label m*1
    :param test_X: data to be tested
    :return: predict of test data
    '''
    m,n=test_X.shape
    y_hat=np.zeros((m))
    for i in range(m):
        expand_test_X=np.full((len(train_X),n),test_X[i,:])
        y_hat[i]=train_y[np.argmax(np.sum(expand_test_X==train_X,axis=1))]
    return y_hat

def estimateGenerationError(m,n,algorithm,runs,test_size):
    testing_error=[]
    for i in range(runs):

        # generate train_X, train_y, and test_X a
        if algorithm=='winnow':
            train_X,train_y=data_generate(m,n,True)
            test_X,test_y=data_generate(test_size,n,True)
        else:
            train_X,train_y=data_generate(m,n)
            test_X,test_y=data_generate(test_size,n)

        # fit and predict test data
        y_hat=eval(algorithm)(train_X,train_y,test_X)
        mistake=len(np.where(y_hat!=test_y)[0])

        # test error rate
        test_error_rate=mistake/test_size
        testing_error.append(test_error_rate)

    return np.mean(testing_error)

def sampleComplexity(algorithm,n_s,max_m,runs,test_size):
    '''
    estimate the minimum number of examples(m) to obtain no more than 10% generalisation error against different dimensions
    :param algorithm: which algorithm you will use: winnow, perceptron, least_square and 1NN
    :param n: dimensions
    :param runs: how many runs
    :return: a list: sample complexity for each dimension
    '''

    sample_complexity_each_n=np.zeros((len(n_s)))
    for index,n in enumerate(n_s):
        m=1
        generalisation_error=0
        while m<max_m:
            generalisation_error=estimateGenerationError(m,n,algorithm,runs,test_size)
            if generalisation_error<=0.1:
                print(f'n: {n}, sample complexity: {m}')
                sample_complexity_each_n[index]=m
                break
            m+=1

    return sample_complexity_each_n

def plotSampleComplexity(sample_complexity,dimensions,algorithm,color,type):
    plt.figure(figsize=(12, 6))
    plt.plot(dimensions,sample_complexity, color,label='Extimate')
    beta_0,beta_1=fitNtoM(dimensions,sample_complexity,type)
    beta_0=np.round(beta_0,4)
    beta_1=np.round(beta_1,4)

    if type=='linear':
        plt.plot(dimensions,beta_0+beta_1*dimensions,label='Fit')
        plt.plot(dimensions,(beta_1+0.2)*dimensions,'--',label=f'{np.round(beta_1+0.2,2)}n')
        plt.plot(dimensions,(beta_1-0.2)*dimensions,'--',label=f'{np.round(beta_1-0.2,2)}n')
        print(f'fit of {algorithm}: {beta_0}+{beta_1}n')
    elif type=='log':
        plt.plot(dimensions, beta_0 + beta_1 * np.log(dimensions), label='Fit')
        plt.plot(dimensions, (beta_1 + 4.2) * np.log(dimensions),'--', label=f'{np.round(beta_1+4.2,2)}log(n)')
        plt.plot(dimensions, (beta_1 - 0.2) * np.log(dimensions),'--', label=f'{np.round(beta_1-0.2,2)}log(n)')
        print(f'fit of {algorithm}: {beta_0}+{beta_1}log(n)')
    elif type=='exp':
        plt.plot(dimensions, np.exp(beta_0+beta_1 * dimensions), label='Fit')
        plt.plot(dimensions, 1.5*np.exp(beta_0+beta_1 * dimensions),'--', label=f'{np.round(beta_1+0.5,2)}exp(n)')
        plt.plot(dimensions, 0.5*np.exp(beta_0+beta_1 * dimensions),'--', label=f'{np.round(beta_1-0.5,2)}exp(n)')
        print(f'fit of {algorithm}: {beta_0}+{beta_1}exp(n)')
    # plt.errorbar(dimensions,sample_complexity,yerr=1.5)
    plt.xlabel('Dimension of sample (n)')
    plt.ylabel('Sample complexity')
    plt.title(f'Sample complexity against n in {algorithm} Algorithm')
    plt.legend()
    plt.show()

# (c) how m grows as a function of n
def fitNtoM(dimensions,sample_complexity,type):
    '''
    estimate how m grows as a function of n as n to inf for each of the four algorithms
    :param dimensions: n dimensions
    :param sample_complexity: minimum m that gain <=10% generalization error in each n
    :param type: type of fit -- log, exp or linear
    :return: parameters
    '''
    if type=='log':
        beta_1,beta_0=np.polyfit(np.log(dimensions),sample_complexity,1)
    elif type=='exp':
        beta_1, beta_0 = np.polyfit(dimensions, np.log(sample_complexity), 1)
    elif type=='linear':
        beta_1,beta_0=np.polyfit(dimensions,sample_complexity,1)

    return beta_0,beta_1


if __name__ == '__main__':
    max_m=10000 # max sample size
    runs=5
    test_size=10000
    n_s = np.arange(1, 101, 1)

    # get sample_complexity for each algorithm
    sample_complexity_perceptron=sampleComplexity('perceptron',n_s,max_m,runs,test_size)
    plotSampleComplexity(sample_complexity_perceptron,n_s,'Perceptron','#2E8B57','linear')
    beta_0,beta_1=fitNtoM(n_s,sample_complexity_perceptron,'linear')
    print(beta_0,beta_1)
    print('perception end')


    print('least square begin')
    sample_complexity_leastq = sampleComplexity('least_square', n_s, max_m, runs,test_size)
    plotSampleComplexity(sample_complexity_leastq, n_s, 'Least Square', '#DDA0DD','linear')
    beta_0, beta_1 = fitNtoM(n_s, sample_complexity_leastq, 'linear')

    print('winnow begin')
    sample_complexity_winnow = sampleComplexity('winnow', n_s, max_m, runs, test_size)
    plotSampleComplexity(sample_complexity_winnow, n_s, 'Winnow', '#F08080','log')
    print('winnow end')

    n_s_1NN = np.arange(1, 16, 1)
    sample_complexity_1NN = sampleComplexity('one_NN', n_s_1NN, max_m, runs, test_size)
    plotSampleComplexity(sample_complexity_1NN, n_s_1NN, '1NN', '#00008B','exp')


    # plt.show()