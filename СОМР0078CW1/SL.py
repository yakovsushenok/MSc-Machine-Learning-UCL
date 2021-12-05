import numpy as np
import math

class Regression:
    '''
    this class is for regression
    '''
    def __init__(self,Sine=False,map=True):
        self.coeff=[]
        self.Sine=Sine
        self.map=map

    def mapping(self,x,k):
        '''
        :param x: x to map
        :param k: dimension
        :return: mapped data in matrix form
        '''
        matrix = np.zeros((len(x), k))  # Declare a 2d matrix to store the data after feature map
        row = 0
        for xi in x:
            column = 0
            while column < k:
                if self.Sine==False:
                    matrix[row, column] = math.pow(xi, column)
                else:
                    matrix[row,column]=np.sin((column+1)*np.pi*xi)
                column += 1
            row += 1

        return matrix


    def getWeight(self,X,y):
        '''
        :param X: X matrix
        :param y: y matrix
        :return: weight-w
        '''
        w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
        self.coeff=w
        return w

    def getPoly(self,x_new):
        '''
        :param x_new: input x
        :return: predict y
        '''
        w = self.coeff
        output=0
        index=0
        for wi in np.nditer(w):
            output=output+wi*x_new**index
            index+=1
        return output

    def MSE(self,pred,y):
        '''
        :param pred: predicted y
        :param y: original y
        :return: mse
        '''
        w=self.coeff
        predict_y = w.dot(pred)
        error = abs(predict_y - y)
        sse = sum([error_i ** 2 for error_i in error])
        return sse / len(pred.T)

class KRR:
    '''
    this class is for kernel ridge regression
    '''
    def __init__(self,):
        self.alpha=[] # use dual representation so we optimize alpha

    def regress(self,gamma,kernel_matrix,y):
        '''
        :param gamma: input gamma
        :param kernel_matrix: k(x_i,x_test)
        :param y: input y
        :return: dual form-alpha
        '''
        self.alpha=np.linalg.inv(kernel_matrix+gamma*kernel_matrix.shape[0]*np.identity(kernel_matrix.shape[0]))@y # @ dot

    def predict(self,kernel):
        '''
        use formula: y_test=sum(alpha_i*k(x_i,x_test))
        :param kernel: kernel matrix
        :return: predicted y
        '''
        return self.alpha@kernel

    def MSE(self,predict,y):
        '''
        :param predict: predicted y
        :param y: actual y
        :return: mse
        '''
        error=predict-y
        sse=sum([error_i**2 for error_i in error])
        return sse/predict.shape[0]
