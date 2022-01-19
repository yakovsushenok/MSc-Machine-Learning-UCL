import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SC import spectralClause

def isotropicGaussian(center,sigma,size):
    '''
    :param center: data centered at (.,.)
    :param sigma: variance
    :param size: number of data needed
    :return: points needed
    '''
    x_s=np.random.normal(center[0],sigma,size)
    y_s=np.random.normal(center[1],sigma,size)

    points=np.zeros((size,2))
    for i in range(size):
        points[i,0]=x_s[i]
        points[i,1]=y_s[i]

    return points

if __name__ == '__main__':
    # set parameter and get points
    size=20
    points_posi=isotropicGaussian([-0.3,-0.3],0.2,size)
    points_nega=isotropicGaussian([0.15,0.15],0.1,size)

    # original data
    plt.figure(figsize=(15,6))
    plt.subplot(2,1,1)
    plt.scatter(points_posi[:,0],points_posi[:,1],c='#F08080',marker='o')
    plt.scatter(points_nega[:,0],points_nega[:,1],c='#F08080',marker='*')
    plt.title('original data')

    # to shuffle data, firstly we have to combine point and their true label
    classes=[1 for i in range(size)]
    classes.extend([-1 for i in range(size)])
    classes=np.array(classes).reshape(2*size,1)
    whole_point=np.append(points_posi,points_nega,axis=0)
    dataset=np.append(classes,whole_point,axis=1)
    np.random.shuffle(dataset)

    # do cluster again and select best c
    data_X = dataset[:, 1:]
    c=[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2]
    error_list=[]
    for c_i in c:
        sc=spectralClause(data_X,c_i)
        weight_matrix = sc.getAdjacency()
        cluster = sc.getClustered()
        error = len(np.where((cluster - dataset[:, 0]) != 0)[0])
        print(f'make {error} mistakes')
        error_list.append(error)
    best_c = c[np.argmin(error_list)]
    print(f'best sigma is: {best_c}')

    # plot clustered data with best c
    sc = spectralClause(data_X, best_c)
    weight_matrix = sc.getAdjacency()
    cluster = sc.getClustered()
    error = len(np.where((cluster != dataset[:, 0]))[0])
    print(f'make {error} mistakes')

    plt.subplot(2,1,2)
    for i in range(2*size):
        if cluster[i] == -1:
            plt.scatter(data_X[i][0], data_X[i][1], c='#F08080',marker='*')
        else:
            plt.scatter(data_X[i][0], data_X[i][1], c='#2E8B57',marker='o')
    plt.title('Spectral clustering algorithm')
    plt.show()

