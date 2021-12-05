# Relevant libraries
import matplotlib.pyplot as plt
import numpy as np

#############################################################################
#                                                                           #
#                           Question 2 (a) (i)                              #
#                                                                           #
#############################################################################
# Setting up the random vectors
np.random.seed(1)
mu, sigma = 0, 0.07 # Our parameters for the noise vector
e = np.random.normal(mu,sigma,30) # The noise vector
X = np.random.uniform(0,1,30) # Our input vector
g = np.sin(2*(np.pi)*X)**2+e # Our output vector

# Creating the plot
x1 = np.linspace(0,1,10000000)
y1 = np.sin(2*(np.pi)*x1)**2
plt.plot(x1, y1, '-r', label=r'$\sin (2\pi x)^2$')
plt.scatter(X,g,label=r'$\sin (2\pi x)^2+e$')
plt.ylim(0,1.17)
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='lower right')
plt.title(label = r'$\sin (2\pi x)^2$ with and without noise')
plt.grid()
plt.show()


#############################################################################
#                                                                           #
#                           Question 2 (a) (ii)                              #
#                                                                           #
#############################################################################
k_list = [2, 5, 10, 14, 18] # Our list of k's

# Creating a for loop to get the coefficient vectors
weight_list = [] # Initialzing the weight list 
for i in range(len(k_list)):
    x = np.ones((30,1))
    for m in range(k_list[i]):
      x = np.c_[x,X**m]
    x= x[:,1:]
    weight_list.append(np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)),x.T), g))

# Creating the plot
x_plot = np.linspace(-1,2,10000)
for i in range(len(k_list)):
  poly = poly_symbolic(weight_list[i])
  y = poly(x_plot)
  plt.plot(x_plot,y,color_list[i],label="k = {}".format(k_list[i]))
plt.scatter(X,g,label='Data')
plt.legend(loc='lower right')
plt.title(label = r"Curves for $k=2,5,10,14,18$")
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.ylim(-0.3,1.2)
plt.xlim(0,1)
plt.show()


#############################################################################
#                                                                           #
#                           Question 2 (b)                                  #
#                                                                           #
#############################################################################

k_list = [i for i in range(1,19)] # Our list of k's

# Creating a for loop to get the coefficient vectors
weight_list = [] # Initialzing the weight list 
for i in range(len(k_list)):
    x = np.ones((30,1))
    for m in range(k_list[i]):
      x = np.c_[x,np.around(X**m,decimals=5)]
    x= x[:,1:]
    weight_list.append(np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)),x.T), g))

# Creating a for loop to get the logarithms of the MSE for all values of k 
mse_arr = []
for i in range(len(k_list)):
  mse_arr.append(np.log(calc_mse(g, X, weight_list[i])))

# Creating the plot for the logarithm of the mean square error for the training data
plt.title(label = 'The logarithm of the mean square error for the training data')
plt.xlabel('k', color='#1C2833')
plt.ylabel( r'$\ln(MSE)$', color='#1C2833')
plt.plot(np.linspace(0,18,18),mse_arr)
plt.xlim(1,18)
plt.show()

#############################################################################
#                                                                           #
#                           Question 2 (c)                                  #
#                                                                           #
#############################################################################
# Creating our test data
np.random.seed(1)
e_test = np.random.normal(mu,sigma,1000)
x_test = np.random.uniform(0,1,1000)
g_test = np.sin(2*(np.pi)*x_test)**2+e_test

# Creating a for loop to get the logarithms of the MSE for all values of k 
mse_arr = []
for i in range(len(k_list)):
  mse_arr.append(np.log(calc_mse(g_test,x_test , weight_list[i])))

# Creating the plot for the logarithm of the mean square error for the test data
plt.title(label = 'The logarithm of the mean square error for the test data')
plt.xlabel('k', color='#1C2833')
plt.ylabel( r'$\ln(MSE)$', color='#1C2833')
plt.plot(np.linspace(0,18,18),mse_arr)
plt.xlim(1,18)
plt.show()


#############################################################################
#                                                                           #
#                           Question 2 (d)                                  #
#                                                                           #
#############################################################################

# We first define our arrays for the average mean square error values over the 100 runs
Mse_avg_tr = np.ravel(np.zeros((18,1)))
Mse_avg_te = np.ravel(np.zeros((18,1)))
# Setting up a loop to do what we did in 2 (b) and (c) 100 times
for run in range(100):
  np.random.seed(run)
  mu, sigma = 0, 0.07
  e = np.random.normal(mu,sigma,30)
  e_test = np.random.normal(mu,sigma,1000)
  X = np.random.uniform(0,1,30)
  x_test = np.random.uniform(0,1,1000)
  g = np.sin(2*(np.pi)*X)**2+e
  g_test = np.sin(2*(np.pi)*x_test)**2+e_test
  k_list = [i for i in range(1,19)] # Our list of k's

  # Creating a for loop to get the coefficient vectors
  weight_list = [] # Initialzing the weight list 
  for i in range(len(k_list)):
      x = np.ones((30,1))
      for m in range(k_list[i]):
        x = np.c_[x,np.around(X**m,decimals=5)]
      x= x[:,1:]
      weight_list.append(np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)),x.T), g))
  mse_arr_tr = np.zeros([])
  mse_arr_te = np.zeros([])
  for i in range(len(k_list)):
    mse_arr_tr = np.append(mse_arr_tr,calc_mse(g, X, weight_list[i]))
    mse_arr_te = np.append(mse_arr_te,calc_mse(g_test, x_test, weight_list[i]))
  Mse_avg_tr+=mse_arr_tr[1:]/100
  Mse_avg_te+=mse_arr_te[1:]/100

# Here we are going to build the plot for logarithm of the average of the mean square error
plt.title(label = 'The logarithm of the average of the mean square error')
plt.xlabel('k', color='#1C2833')
plt.ylabel( r'$\ln(MSE)$', color='#1C2833')
plt.plot(np.linspace(0,18,18), np.log(Mse_avg_tr),label = "Train")
plt.plot(np.linspace(0,18,18), np.log(Mse_avg_te), label = "Test")
plt.legend(loc='lower right')
plt.xlim(1,18)
plt.show()