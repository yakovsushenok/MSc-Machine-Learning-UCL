import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

class MethodError(Exception):
    pass

class Kernel:
    '''
    This class aims to calculate different kinds of kernels
    '''
    def __init__(self, method, dimension):
        self.method = method   # method stands for type of kernel. Method can be either "poly" (polynomial kernel) or "gaussian" (gaussian kernel)
        self.dimension = dimension # dimension stands for the type of parameters. For the polynomial kernel it's d and for gaussian kernel it's c.

    def getFeature(self, X1, X2):
        '''
        This method uses polynomial kernel and gaussian kernel to transform data
        if one did not declare either of these two method then it will raise an error
        Args:
                X1 - Matrix 1
                X2 - Matrix 2
        Returns:
                Kernel matrix which is calculated by the polynomial or gaussian kernel
        '''
        # Calulating the polynomial kernel
        if self.method == 'poly':
            return X1.dot(X2.T)**self.dimension
        
        elif self.method == 'gaussian':
            # Here we assume that matrices X1 and X2 are different dimensions. 
            # So assume that X1 is of dimension n1*m and X2 is of dimension n2*m. We should return a kernel matrix which is of dimension n1*n2 
            square_sum_x1_row = np.sum(X1**2, axis=1)
            square_sum_x2_row = np.sum(X2**2, axis=1)
            num_x1_data = X1.shape[0]
            num_x2_data = X2.shape[0]
            x1i = square_sum_x1_row.reshape((num_x1_data, 1)) # x1i -- n1*1
            x2j = square_sum_x2_row.reshape((1, num_x2_data)) # x2j -- 1*n2

            # x1j+x2j -- n1*n2

            return np.exp(-self.dimension * (x1i + x2j - 2 * X1.dot(X2.T)))
        
        else:
            
            raise MethodError('Kernel method should be either poly or gaussian')

class Multiclass:
    """
    This class trains and predicts the online multi-class perceptron model. It has two ways of generalizing to k-classes - "One vs Rest" (OvR) and "One vs One" (OvO).
    """
    def __init__(self, alphas, mapped_X, y, num_class, epoch):
        
        self.alphas = alphas          # Our classifiers
        self.mapped_X = mapped_X      # Kernel Matrix
        self.y = y                    # Original Lables
        self.num_class = num_class    # Number of classes 
        self.epoch = epoch            # Number of epochs
        self.num_classifiers = int(self.num_class*(self.num_class-1)/2)      # Number of clasifiers (used for OvO method)
        self.class_combinations = list(combinations(range(num_class), 2))   # Class combinations like 0 - 1, 0 - 2, 0 - 3......, 0 - 9.1 - 2, 1 - 3..., 1 - 9, 2 - 3, 2 - 4, 2 - 5..., 2 - 9,...8 - 9

    def fit(self):
        """
        This functions trains the perceptron classifiers using the "One vs Rest" method. Each class has one classifier.
        Returns:
                Trained classifiers (alphas)
        """
        # number of samples
        n_sample = len(self.mapped_X)
        
        for i in range(self.epoch):
            
            # here the classifiers (alpha) matrix takes the dimension num_class*n_sample size and the kernel matrix takes dimension n_sample*n_sample size
            for j in range(n_sample): 
                
                # to do t th prediction, we use alpha matrix dot product the t th column of kernel matrix
                pred = np.argmax(np.dot(self.alphas, self.mapped_X[:, j]))
                
                # update
                if pred != self.y[j]: 
                    
                    # if we predict incorrectly, we subtract 1 from the value of the classifier's entry corresponding to the incorrectly predicted class and sample index and add 1 to the true class' classifier
                    self.alphas[pred, j] -= 1 
                    self.alphas[self.y[j], j] += 1
                    
        return self.alphas

    def OVOfit(self):
        # classifiers like 0-1,0-2,0-3......,0-9.1-2,1-3...,1-9,2-3,2-4,2-5...,2-9,...8-9
        """
        This functions trains the perceptron classifiers using the "One vs One" method. Each class pair combination has a classifier. This means in total we will have 45 classifiers for 10 classes.
        """
        # number of samples
        n_sample = len(self.mapped_X)
        for i in range(self.epoch):
            
            # here the classifiers (alpha) matrix is num_classifier*n_sample size, kernel matrix is n_sample*n_sample size
            for j in range(n_sample): # Iterating through the whole dataset by each sample
                
                # We first calculate the dot products of each classifier with the relevant column of the kernel matrix
                confident = np.dot(self.alphas, self.mapped_X[:, j])
                
                # We convert the confindent array to have values 1 if the product is positive and -1 otherwise
                binary_prediction = [1 if con > 0 else -1 for con in confident]
                
                # We use location_predicted_class as a "helper array" to get the index of the predicted class later when we want to predict the specific class. 
                location_predicted_class = np.clip(binary_prediction, a_min=None, a_max=0).astype(int)
                
                # We are now going to get the prediction of each classifier
                prediction = np.zeros((self.num_classifiers)).astype(int)
                for classifier in range(len(confident)):
                    """
                    For each classifier, we have a pair of classes which the classifier tries to classify each time, which are stored in the class_combination array as 2 dimensional arrays. At each 
                    iteration of this loop, we call the class_combinations array and the "helper array" located_predicted_class. If the classifier predicts a positive number, then the first
                    class of the respective class pair is the prediction, and the second class if the classifier predicts a negative number.
                    """
                    predicted = self.class_combinations[classifier][location_predicted_class[classifier]]
                    prediction[classifier] = predicted
                """
                Here we do the update step in case of an incorrect prediction.
                Suppose we have a classifier for the pair [2,5]. If the classifier predicts a positive number, then the prediction will be 2, and 5 otherwise. Now, consider the following 3 cases:
                
                1. The true label is 2
                2. The true label is 5
                3. The true label is a, where a is some number in the interval 0,...,9 and is not 2 or 5
                
                In the first case, if we predict a number which is not 2, we add 1 to the elements of the classifier
                In the second case, if we predict a number which is not 5, we subtract 1 from the elements of the classifier
                In the third case, if we predict any number, we don't update the classifier since it "didn't do anything wrong".
                """
                for index, classify in enumerate(self.class_combinations):
                    
                    yhat_i = prediction[index]
                    
                    if self.y[j] == classify[0] and yhat_i != self.y[j]: # Analogous to the 1st case
                        self.alphas[index,j]+=1
                    
                    elif self.y[j] == classify[1] and yhat_i != self.y[j]: # Analogous to the 2nd case
                        self.alphas[index,j] -= 1
        
        return self.alphas


    def predict(self, X, OVO=False):
        """
        This function predicts values for the elements of the input matrix X. Predictions are mainly used to evaluate the model performance. 
        """
        n_sample = X.shape[1]
        pred=[]
        
        for i in range(n_sample):
            
            if OVO is False: # Predictions using the OvR method
                
                predict = np.argmax(np.dot(self.alphas, X[:, i]))
                pred.append(predict)
            
            else:  # Predictions using the OvO method. A sample is predicted using the method described in the function "OvOfit"
                
                confident = np.dot(self.alphas, X[:, i])
                binary_prediction = [1 if con > 0 else -1 for con in confident]
                location_predicted_class = np.clip(binary_prediction, a_min=None, a_max=0).astype(int)
                prediction = np.zeros((self.num_class, 1))
                
                for classifier in range(len(confident)):
                    predicted = self.class_combinations[classifier][location_predicted_class[classifier]]
                    prediction[predicted] += 1
                prediction = np.argmax(prediction)
                pred.append(prediction)
        
        return pred


