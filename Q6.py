import numpy as np
import matplotlib.pyplot as plt
import kNN
from matplotlib.colors import ListedColormap

def Bernoulli(p,size):
    '''
    p: probability of head
    size: number of random variables
    '''
    rvs = np.array([])
    for i in range(0,size):
        if np.random.rand() <= p:
            a=1
            rvs = np.append(rvs,a)
        else:
            a=0
            rvs = np.append(rvs,a)
    return rvs

if __name__ == '__main__':
    v=3
    knn=kNN.kNN(v)
    s_x,s_y=knn.sample_centers(S=100)
    y_label = Bernoulli(0.5, 100)

    sx_label_1 = []
    sx_label_0 = []
    sy_label_1 = []
    sy_label_0 = []
    for i in range(100):
        if y_label[i] == 1:
            sx_label_1.append(s_x[i])
            sy_label_1.append(s_y[i])
        else:
            sx_label_0.append(s_x[i])
            sy_label_0.append(s_y[i])

    # s_x,s_y into a tuple for further analysis
    X=[(s_x[i],s_y[i]) for i in range(100)]

    #=================== Q6 =========================
    clf=kNN.kNN(v)
    clf.fit(X,y_label)
    predict=clf.predict(X)

    # plot decision boundary
    xp = np.linspace(0, 1, 100)  
    yp = np.linspace(0, 1, 100)
    x1, y1 = np.meshgrid(xp, yp)
    xy = np.c_[x1.ravel(), y1.ravel()]
    y_pred=clf.predict(xy).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff'])

    plt.figure(figsize=(12, 6))
    plt.contourf(x1, y1, y_pred, alpha=0.3, cmap=custom_cmap)
    p1=plt.scatter(sx_label_1, sy_label_1)
    p2=plt.scatter(sx_label_0, sy_label_0)
    plt.legend([p1, p2], ['1','0'], loc='upper right')
    plt.show()
