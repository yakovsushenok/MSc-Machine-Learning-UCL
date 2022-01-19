from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SC import spectralClause

def read_dat(path):
    '''
    :param path: relative path root for dat file
    :return: dataset
    '''
    dat = pd.read_csv(path, sep='\s+', header=None).to_numpy()
    return dat

if __name__ == '__main__':
    file_path = Path('./twomoons.dat')
    dataset = read_dat(file_path)
    data_X=dataset[:,1:]
    print(data_X)
    plt.figure(figsize=(12,6))
    # plot the original data
    for x in data_X:
        plt.scatter(x[0],x[1],c='b')
    plt.show()

    num_x=data_X.shape[0] # number of data

    # range of c
    index=np.arange(-10,10.1,0.1).astype(np.float64)
    sigma=[2**i for i in index]

    # Start clustering in each c
    error_for_each_sigma=[]
    for sig in range(len(sigma)):
        c=sigma[sig]
        sc=spectralClause(data_X,c)
        weight_matrix=sc.getAdjacency()
        cluster=sc.getClustered()
        error=len(np.where((cluster - dataset[:,0]) != 0)[0])
        print(f'2^{np.round(index[sig],2)}: make {error} mistakes')
        error_for_each_sigma.append(error)

    # select best c
    best_sigma=sigma[np.argmin(error_for_each_sigma)]
    print(f'best sigma is: {best_sigma}')

    # plot clustered data with best c
    sc = spectralClause(data_X, best_sigma)
    weight_matrix = sc.getAdjacency()
    cluster = sc.getClustered()
    error = len(np.where((cluster != dataset[:, 0]))[0])
    print(f'make {error} mistakes')

    plt.figure(figsize=(12, 6))
    for i in range(num_x):
        if cluster[i] == -1:
            plt.scatter(data_X[i][0], data_X[i][1], c='b')
        else:
            plt.scatter(data_X[i][0], data_X[i][1], c='r')
    plt.show()




