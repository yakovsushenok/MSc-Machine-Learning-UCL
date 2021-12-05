import numpy as np
import matplotlib.pyplot as plt
import kNN
from tqdm import tqdm
import time
import numba as nb

def Bernoulli(p,size,single=False):
    '''
    p: probability of head
    size: number of random variables
    '''
    rvs = np.array([])
    for i in range(0,size):
        if np.random.rand() <= p:
            a=1
            if single==True:
                rvs=a
            else:
                rvs = np.append(rvs,a)
        else:
            a=0
            if single == True:
                rvs = a
            else:
                rvs = np.append(rvs, a)
    return rvs

def sample_h_pH(knn):
    '''
    :param knn: object knn
    :return: sampled X in unit square and y from U[0,1]
    '''
    s_x, s_y = knn.sample_centers(S=100)
    y_label = Bernoulli(0.5, 100)
    X=np.zeros((len(s_x),2),dtype=float)

    for i in range(len(s_x)):
        X[i][0]=s_x[i]
        X[i][1] = s_y[i]
    return X,y_label


# @nb.jit
def data_from_phxy(knn,p_heads,X,iter):
    '''
    :param knn: object knn
    :param p_heads: prob(heads)
    :param X: centers
    :param iter: numbers of label to be generated
    :return: label sampled from ph(x,y)
    '''
    label=[]
    for i in range(iter):
        if np.random.rand() <= p_heads:
            single_label = knn.predict(X[i],3,single=True)
            label.append(single_label)
        else:
            single_label = Bernoulli(0.5, 1, single=True)
            label.append(single_label)
    return label

def getX(x,y):
    '''
    :param x: input x
    :param y: input y
    :return: return matrix of points (n*d)
    '''
    X = np.zeros((len(x), 2), dtype=float)
    for i in range(len(x)):
        X[i][0] = x[i]
        X[i][1] = y[i]
    return X


def build_knn(knn,p_heads,num,k):
    '''
    :param knn: object knn
    :param p_heads: prob(H)
    :return: build knn model(training data label, training x, training y)
    '''
    training_x, training_y = knn.sample_centers(S=num)
    training_X = getX(training_x,training_y,num)
    training_label=data_from_phxy(knn,p_heads,training_X,num)
    predict=knn.predict(training_X,k)
    return training_label,training_X,training_y,predict


def testing_knn(knn,training_X,p_heads,k,training_label):
    '''
    :param knn: object knn
    :param training_X: X as training data
    :param p_heads: prob(H)
    :param k: k nearest neighbour
    :param training_label: training data's label
    :return: label of testing data, predicted label of testing data
    '''
    test_x, test_y = knn.sample_centers(S=1000)
    testing_X = getX(test_x,test_y,1000)
    testing_label=data_from_phxy(knn,p_heads,testing_X,1000)
    knn.fit(training_X, training_label)
    predict = knn.predict(testing_X,k)

    return predict,testing_label

# @nb.jit
def getRunningError(predict,check):
    '''
    :param predict: predicted label
    :param check: actual label
    :return: number of difference between predicted and actual
    '''
    difference = 0
    iter=len(predict)
    for j in range(iter):
        if predict[j] != check[j]:
            difference += 1
    return difference

if __name__ == '__main__':

    start = time.perf_counter()
    klist=np.arange(1,49,1)
    generalisation_error=[]
    mlist=[100]
    m=np.arange(500,4500,500)
    mlist.extend(m)


    error_opt_k = []
    for m in tqdm(range(len(mlist))):
        num=mlist[m]
        opt_k=[]
        for i in tqdm(range(100)):
            running_error=[]
            for k in klist:
                knn = kNN.kNN(3)

                # sample a h from pH
                X_sample, y_label = sample_h_pH(knn)
                knn.fit(X_sample, y_label)

                # Build a kNN model with m training points sampled from ph(x,y)
                p_heads = 0.8
                training_label, training_X, training_y,training_predict = build_knn(knn, p_heads,num,k)


                # Run kNN estimate generalisation error
                predict, testing_label = testing_knn(knn, training_X, p_heads, k, training_label)

                running_error.append(getRunningError(predict, testing_label) / 1000)
            k_choose = np.argsort(running_error)[0]
            opt_k.append(klist[k_choose])
        print(opt_k)
        error_opt_k.append(np.mean(opt_k))

    print(error_opt_k)

    plt.figure(figsize=(12,6))
    plt.plot(mlist,error_opt_k)
    plt.xlabel('m')
    plt.ylabel('average optimal k')
    plt.title('Determine the optimal k')
    plt.show()

