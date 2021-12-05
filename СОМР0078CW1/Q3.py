# Relevant libraries
import matplotlib.pyplot as plt
import numpy as np

#############################################################################
#                                                                           #
#                           Building relevant functions                     #
#                                                                           #
#############################################################################


def symbolic_sin(coef_array):
    """
    This function returns a sine expression which can then be evaluated for a given x
    Args:
          coeff_array - the array of the coefficients.
    Output:
          A sine expression.
    """
    return lambda x: sum(c*np.sin(x*(i+1)*np.pi) for i, c in enumerate(coef_array))

def gk(k,x):
    """
    This function returns sin(k*\pi*x) for some k and x
    Args:
          k - Degree of the polynomial expansion.
          x - The input of the function sin(k*\pi*x).
    Output:
          sin(k*\pi*x)
    """
    g = np.sin(k*(np.pi)*x).reshape(-1,1)
    return g

def sin_basis(k,x):
    """
    This function performs a sine basis expansion
    Args:
          k - Degree of the polynomial expansion.
          x - The data matrix.
    Output:
          A modified data matrix.
    """
    xnew = np.around(gk(1,x),decimals=7)
    for i in range(1,k+1):
        xnew = np.append(xnew,gk(i,x),axis=1)
    return xnew[:,1:]

def calc_mse_sin(true_val,x_val,coef):
    """
    This function calculates the mean square error of the predictions of a fitted
    curve/line with the sine basis expansion
    Args:
            true_val - The target vector, i.e., the "ground truth".
            x_val - The input values for which we want to predict the y values of.
            coef - The coefficient vector.
    Output:
            The mean square error.
    """
    poly = symbolic_sin(coef)
    y_pred = poly(x_val)

    return ((true_val - y_pred)**2).mean()


#############################################################################
#                                                                           #
#                           Building the first plot                         #
#                                                                           #
#############################################################################

  
def polynomial_sse_sin(randomSeed):
    """
    This function calculates the mean square error of the curves with k = 1,...,18 sine basis expansion on the training data.
    Args:
            ranomdSeed - The number of the random seed. It is used mainly to generate new random data when iterating 100 times for the averaging of the MSE's.
    Output:
            The mean square error list for all k's.
    """
    np.random.seed(randomSeed)
    mu, sigma = 0, 0.07
    e = np.random.normal(mu,sigma,30)
    x = np.random.uniform(0,1,30)
    g = np.sin(2*(np.pi)*x)**2+e
    mse_array=[]
    for i in range(1,19):
            xnew = sin_basis(i,x)
            w = np.dot(np.linalg.inv(np.dot(xnew.transpose(),xnew)),np.dot(xnew.transpose(),g))     
            mse_array.append(calc_mse_sin(g,x,w))
    return mse_array

# Producing the plot with the logarithm of the mean square error for training data

plt.plot(np.linspace(0,18,18),np.log(polynomial_sse_sin(6)))
plt.title(label = r'The logarithm of the mean square error for training data')
plt.xlabel('k', color='#1C2833')
plt.ylabel( r'$\ln(MSE)$', color='#1C2833')
plt.xlim(1,18)
plt.show()

#############################################################################
#                                                                           #
#                           Building the second plot                        #
#                                                                           #
#############################################################################



def polynomial_sse_sin_test(randomSeed):
    """
    This function calculates the mean square error of the curves with k = 1,...,18 sine basis expansion on the test data with the parameters obtained from the train data.
    Args:
            ranomdSeed - The number of the random seed. It is used mainly to generate new random data when iterating 100 times for the averaging of the MSE's.
    Output:
            The mean square error list for all k's.
    """
    np.random.seed(randomSeed)
    mu, sigma = 0, 0.07
    e = np.random.normal(mu,sigma,30)
    e_test = np.random.normal(mu,sigma,1000)
    x = np.random.uniform(0,1,30)
    x_test = np.random.uniform(0,1,1000)
    g = np.sin(2*(np.pi)*x)**2+e
    g_test = np.sin(2*(np.pi)*x_test)**2+e_test
    mse_array=[]
    for i in range(1,19):
            xnew = sin_basis(i,x)
            w = np.dot(np.linalg.inv(np.dot(xnew.transpose(),xnew)),np.dot(xnew.transpose(),g))     
            mse_array.append(calc_mse_sin(g_test,x_test,w))
    return mse_array

# Producing the plot with the logarithm of the mean square error for test data

plt.plot(np.linspace(0,18,18),np.log(np.array(polynomial_sse_sin_test(2))))
plt.title(label = r'The logarithm of the mean square error for test data')
plt.xlabel('k', color='#1C2833')
plt.ylabel( r'$\ln(MSE)$', color='#1C2833')
plt.xlim(1,18)
plt.show()


#############################################################################
#                                                                           #
#                           Building the third plot                         #
#                                                                           #
#############################################################################

# Here we are calculating the lists which will contain the average MSE's over 100 runs 
mse_tr = np.ravel(polynomial_sse_sin(0))/100
mse_te = np.ravel(polynomial_sse_sin_test(0))/100
for i in range(1,100):
    mse_tr += np.ravel(polynomial_sse_sin(i))/100
    mse_te += np.ravel(polynomial_sse_sin_test(i))/100

# Here we are building the plot for the logarithm of the average of the mean square error
  
plt.title(label = 'The logarithm of the average of the mean square error')
plt.xlabel('k', color='#1C2833')
plt.ylabel( r'$\ln(MSE)$', color='#1C2833')
plt.plot(np.linspace(0,18,18), np.log(mse_tr),label = "Train")
plt.plot(np.linspace(0,18,18), np.log(mse_te), label = "Test")
plt.legend(loc='lower right')
plt.xlim(1,18)
plt.show()
