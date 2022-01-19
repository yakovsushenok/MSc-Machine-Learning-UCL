import pandas as pd
import numpy as np

class spectralClause:
    def __init__(self,data_X,c):
        '''
        :param data_X: data
        :param c: c
        '''
        self.X=data_X
        self.c=c

    def getAdjacency(self):
        # assume X1 is n1*m, X2 is n2*m, we should return a matrix with n1*n2
        square_sum_x1_row = np.sum(self.X ** 2, axis=1)
        square_sum_x2_row = np.sum(self.X ** 2, axis=1)
        num_x1_data = self.X.shape[0]
        num_x2_data = self.X.shape[0]
        x1i = square_sum_x1_row.reshape((num_x1_data, 1))  # x1i -- n1*1
        x2j = square_sum_x2_row.reshape((1, num_x2_data))  # x2j -- 1*n2

        # x1j+x2j -- n1*n2

        self.weight=np.exp(-self.c * (x1i + x2j - 2*self.X.dot(self.X.T)))

        return self.weight


    def getClustered(self):
        # set D
        D=np.diag(np.sum(self.weight,axis=1))


        # get graph laplacian L
        L=D-self.weight
        eigenvalues,eigenvectors=np.linalg.eig(L)
        sort_eigenvalues=np.sort(eigenvalues)
        print(f'second smallest eig is {sort_eigenvalues[1]}')
        sort_index=np.argsort(eigenvalues)

        # get second smallest eigenvalue and its eigenvector
        v2=eigenvectors[:,sort_index[1]]
        print('v2: ',v2)

        cluster=[1 if np.sign(v2[i])!=-1 else -1 for i in range(len(v2))]

        return cluster

