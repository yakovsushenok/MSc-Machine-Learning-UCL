import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from SC import spectralClause

def read_dat(path):
    '''
    read dat file
    :param path: relative file path
    :return: data
    '''
    dat = pd.read_csv(path, sep='\s+', header=None).to_numpy()
    return dat

if __name__ == '__main__':
    # preparation and choose data with label=1,3
    file_path=Path('./dtrain123.dat')
    dataset=read_dat(file_path)

    y_ori=dataset[:,0]

    index_to_delete=np.where(y_ori==2)
    for i in range(len(index_to_delete))[::-1]:
        dataset=np.delete(dataset,index_to_delete[i],axis=0)

    X = dataset[:, 1:]
    dimension_x = X.shape[1]
    y=dataset[:,0]

    # change class to 1 and -1
    y=[1 if i==1 else -1 for i in y]

    # set range of c
    c_s=np.arange(0,0.1,0.002)

    # calculate CP(c) for each c
    CP_cs=[]
    points=np.zeros((len(c_s),2)) # for scatter plot
    for c in range(len(c_s)):
        sc=spectralClause(X,c_s[c])
        weight_matrix=sc.getAdjacency()
        print(weight_matrix)
        cluster=sc.getClustered()
        error=0
        for i in range(len(y)):
            if y[i]!=cluster[i]:
                error+=1
        print(f'{np.round(c_s[c],3)}: make {error} mistakes')
        true=len(y)-error
        CP_c=np.max([true,error])/len(y)
        CP_cs.append(CP_c)

        # points
        points[c,0]=c_s[c]
        points[c,1]=CP_c
    # print(CP_cs)

    plt.figure(figsize=(20,6))
    plt.plot(c_s,CP_cs,'#F08080',)
    for i in range(len(c_s)):
        plt.scatter(points[i,0],points[i,1],c='#2E8B57',marker='*')
    plt.xlabel('Parameter of the Gaussian')
    plt.xticks(c_s,rotation=60)
    plt.ylabel('Correct cluster percentage')

    # ---------------------------------------------
    plt.xlim(0, None) # set 0 as original point
    # ---------------------------------------------

    plt.title('Model selection of parameters')
    plt.show()


