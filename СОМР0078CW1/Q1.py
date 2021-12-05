# Relevant libraries
import matplotlib.pyplot as plt
import numpy as np

#############################################################################
#                                                                           #
#                           Question 1 (a) - (b)                            #
#                                                                           #
#############################################################################

X = np.array([[1], [2], [3], [4]]) # Assigning the appropriate values to our data matrix
yt = np.array([[3], [2], [0], [5]]) # Target vector y
k_list = [1,2,3,4] # List of values of the dimensions of our expansions
weight_list = [] # Creating list for the coefficient vectors of the polynomials
# We will now iterate through every dimension k=1,2,3,4 and find the coefficients associated with each k
for i in range(len(k_list)):
    x = np.ones((4,1))
    for m in range(k_list[i]):
      x = np.c_[x,X**m]
    x= x[:,1:]
    weight_list.append(np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)),x.T), yt))

# Now that we have found the coefficients, we can plot the curves. 
def poly_symbolic(coef_array):
  """
  This function creates a polynomial object which can be later evaluated
  Args:
          coef_array - Array of coefficients of the polynomial
  Output:
          Polynomial object. For example if coef =[1,5] then it returns
          y = 1 + 5*x. 
  """
  return lambda x: sum(c*x**i for i, c in enumerate(coef_array))

color_list = ['blue','yellow','cyan','magenta','green','black'] 

# Creating the plot
x_plot = np.linspace(-1,6,1000)
for i in range(len(k_list)):
  poly = poly_symbolic(weight_list[i])
  y = poly(x_plot)
  plt.plot(x_plot,y,color_list[i],label="k = {}".format(k_list[i]))
plt.scatter(X, yt, label = "Data")
plt.legend(loc='lower right')
plt.xlabel('x', color='#1C2833')
plt.ylabel('Predicted Values', color='#1C2833')
plt.title(label = "Polynomial Fitted Curves for k = 1,2,3,4")
plt.ylim(-6,12)
plt.xlim(-0.5,5)
plt.show()


#############################################################################
#                                                                           #
#                           Question 1 (c)                                  #
#                                                                           #
#############################################################################

def calc_mse(true_val,x_val,coef):
  """
  This function calculates the mean square error of the predictions of a fitted
  curve/line.
  Args:
          true_val - The target vector, i.e., the "ground truth".
          x_val - The input values for which we want to predict the y values of.
          coef - The coefficient vector
  Output:
          The mean square error
  """
  poly = poly_symbolic(coef)
  y_pred = poly(x_val)
  return ((true_val - y_pred)**2).mean()

# We will now iterate through each polynomial and calculate the respective MSE
for i in range(len(k_list)):
  print("For k={}, MSE = {}".format(k_list[i], calc_mse(yt, X, weight_list[i])))