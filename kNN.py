import numpy as np
from collections import Counter
from numba import jit

def Euclidean_dis(x1,x2):
    # x1=(x,y)
    # x2=(x,y)
    # x belongs to unit square [0,1]^2
    return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

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


class kNN:
    '''
    this class is for kNN algorithm
    '''
    def __init__(self,k):
        self.k=k

    def sample_centers(self,S):
        '''
        :param S: number of centers(samples) would like to be sampled
        :return: point x
        '''
        s_x = np.random.uniform(0, 1, S)
        s_y = np.random.uniform(0, 1, S)
        return s_x,s_y

    def fit(self,X,y):
        '''
        :param X: matrix of X as training x
        :param y: matrix of y as training y
        :return: just put them in self
        '''
        self.X_train=X
        self.y_train=y

    def predict(self,X,k,single=False):
        '''
        :param X: matrix X as testing x
        :param k: k nearest neighbourhood
        :param single: if we want to do kNN on single point
        :return:
        '''
        if single==True:
            distance = -2 * self.X_train @ X.T + X @ X + np.sum(self.X_train ** 2, axis=1)
            distance = np.sqrt(abs(distance))
            k_choose = np.argsort(distance)[:k]
            k_nearest_labels = [self.y_train[i] for i in k_choose]
            most_common = Counter(k_nearest_labels).most_common(1)
            if k%2==0 and most_common[0][1]==k/2:
                return Bernoulli(0.5,1,True)
            else:
                return int(most_common[0][0])
        else:
            distances = -2 * self.X_train @ X.T + np.sum(X ** 2, axis=1) + np.sum(self.X_train ** 2, axis=1)[:, np.newaxis]
            distances[distances < 0] = 0
            distances = distances ** .5
            indices = np.argsort(distances, 0) # return the nearest k each column
            k_choose = indices[0:k, :]

            k_nearest_labels = np.zeros((k_choose.shape))
            for i in range(k_nearest_labels.shape[0]):
                for j in range(k_nearest_labels.shape[1]):
                    k_nearest_labels[i][j] = self.y_train[k_choose[i][j]]

            predict=[]
            for i in range(k_nearest_labels.shape[1]):
                most_common = Counter(k_nearest_labels[:, i]).most_common(1)
                # corner case
                if k % 2 == 0 and most_common[0][1] == k / 2:
                    predict.append(Bernoulli(0.5, 1, True))
                else:
                    predict.append(int(most_common[0][0]))
            return predict
