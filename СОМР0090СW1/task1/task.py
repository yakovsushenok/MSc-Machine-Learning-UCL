import tensorflow as tf
import time

# Polynomial function
def polynomial_fun(w, x):
    """
    This function returns the value of a polynomial
    Args:
            w - Vector of the coefficients of length M + 1
            x - Scalar.
    Returns:
            y = sum^M_{m=0}w_m*x^m, where M = degree of the polynomial 
    """
    x = tf.Variable(tf.repeat(x, repeats = w.shape[0]))
    for i in range(w.shape[0]):
        x[i].assign(x[i]**i)
    y = tf.tensordot(x, w, 1)
    return y

# Polynomial Least Squares
def fit_polynomial_ls(dataset, M):
    """
    This function calculates the coefficients obtained by doing polynomial regression.
    Args:
            dataset - N pairs of predictor and target values
            M - The dimension of the polynomial.
    Returns:
            w = (x^Tx)^{-1}x^Ty, where x is the basis expansion matrix and y is the target vector 
    """
    # We first tranform our data
    x = tf.Variable(tf.gather(dataset, 0, axis=1))
    x = tf.reshape(x, [x.shape[0], 1])

    y = tf.Variable(tf.gather(dataset, 1, axis=1))
    y = tf.reshape(y, [x.shape[0], 1])
    
    # We now calculate the basis expansion matrix 
    xnew = tf.Variable(tf.ones(x.shape[0]))
    xnew = tf.reshape(xnew, [x.shape[0], 1])
    for k in range(1,M+1):
        xnew = tf.concat(axis = 1, values = [xnew,x**k]) 
    # We finally calculate the coefficients
    w = tf.tensordot(tf.linalg.inv(tf.tensordot(tf.transpose(xnew),xnew,1)),tf.tensordot(tf.transpose(xnew),y,1),1)
    return w

# Stochastic SGD
def fit_polynomial_sgd(dataset, M, learning_rate, batch_size):
    
    def polynomial_fun_vector(x, M): 
      # The same polynomial function as before but now extended for vectors
      # M here means that the degree of the polynomial is M-1
      
      # We first tranform our data
      x = tf.reshape(x, [x.shape[0],1])
      
      # We now calculate the basis expansion matrix 
      xnew = tf.Variable(tf.ones(x.shape[0]))
      xnew = tf.reshape(xnew, [x.shape[0],1])
      for k in range(1,M):
          xnew = tf.concat(axis = 1, values = [xnew,x**k]) 

      
      y = tf.tensordot(xnew, weights, 1)
      return y

    def loss_rmse(y_pred, y_true):
        loss = tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))
        return loss
   
    
    M = M+1
    # Batched Dataset (we will have 4 minibatches)
    splits = int(dataset.shape[0]/batch_size)
    batched_dataset = []
    for i in tf.split(dataset, num_or_size_splits = splits, axis=0):
        batched_dataset.append(i)

    # Initializing our weights
    weights = tf.Variable(tf.random.normal([M,1]))

    optimal_weights =[]

    epochs = 10000
    for epoch in range(epochs):

        # Compute loss within Gradient Tape 
        for batch in batched_dataset:
            with tf.GradientTape() as tape:
                y_predicted = polynomial_fun_vector(batch[:,:-1], M)
                loss = loss_rmse(y_predicted, batch[:,-1])
           
            # Compute the gradients
            gradients = tape.gradient(loss, [weights])
            
            
            # Adjust the weights
            weights.assign_sub(gradients[0]*learning_rate)


            #print(f'Predicted {y_predicted}: Ground Truth: {batch[:,-1]}')
        optimal_weights.append([loss,tf.keras.backend.get_value(weights)])
        # Print output
        print(f"Epoch {epoch + 1}: Root Mean Square Error: {tf.keras.backend.get_value(loss)}: ") #Weights: {tf.keras.backend.get_value(weights)}
        if tf.keras.backend.get_value(loss) < 10.0:
          break
        else:
          continue
        

    column0 = [row[0] for row in optimal_weights]
    val, idx = min((val, idx) for (idx, val) in enumerate(column0))


    return optimal_weights[idx][1]




train_set_size = 100
test_set_size = 50

# Generating our data
rng = tf.random.Generator.from_seed(1)
train_x = rng.uniform([train_set_size, 1], minval=-20.0, maxval=20.0, dtype=tf.dtypes.float32)
test_x = rng.uniform([test_set_size, 1], minval=-20.0, maxval=20.0, dtype=tf.dtypes.float32)

y = tf.Variable(tf.zeros([train_set_size,1]))
yt = tf.Variable(tf.zeros([test_set_size,1]))
noise_train = rng.normal([train_set_size, 1], mean = 0.0, stddev= 0.2)
noise_test = rng.normal([test_set_size, 1], mean = 0.0, stddev= 0.2)

# Generating y for training with polynomial_fun
for i in range(train_set_size):
    y[i].assign(tf.reshape(polynomial_fun(tf.Variable([1., 2, 3, 4]),train_x[i]),[1]))

# Our train target vector
t_train = tf.add(y, noise_train)

# Training set
training_set = tf.concat(axis = 1, values = [train_x, t_train])

for i in range(test_set_size):
    yt[i].assign(tf.reshape(polynomial_fun(tf.Variable([1., 2, 3, 4]),test_x[i]),[1]))

# Our test target vector
t_test = tf.add(yt,noise_test)

# Test set
test_set = tf.concat(axis = 1, values = [test_x, t_test])







# We first compute the optimal weight vector using the training set
optimal_w = fit_polynomial_ls(training_set, 3)

preds_train = tf.Variable(tf.zeros([train_set_size,1]))
preds_test = tf.Variable(tf.zeros([test_set_size,1]))

# We now predict y^{hat} for train set
for i in range(train_set_size):
    preds_train[i].assign(polynomial_fun(optimal_w,train_x[i]))

# We now predict y^{hat} for test set
for i in range(test_set_size):
    preds_test[i].assign(polynomial_fun(optimal_w, test_x[i]))



print("\n---------------------------------------------------------------------------------------------------------------------------------------------")
print("\nReport, using printed messages, the mean (and standard deviation) in difference a) between the observed training data and the underlying “true” polynomial curve; and b) between the “LS-predicted” values and the underlying “true” polynomial curve\n")

# Calculating the mean and standard deviation in difference between the observed data and the underlying “true” polynomial curve
# train set
mean_training_observed = tf.math.reduce_mean(tf.abs(tf.subtract(y,t_train)))
std_training_observed = tf.math.reduce_std(tf.abs(tf.subtract(y,t_train)))
print(f"Training set: Difference between observed and \"true\" data: Mean: {tf.keras.backend.get_value(mean_training_observed)}. Standard Deviation: {tf.keras.backend.get_value(std_training_observed)}\n")
# test set
mean_test_observed = tf.math.reduce_mean(tf.abs(tf.subtract(yt,t_test)))
std_test_observed = tf.math.reduce_std(tf.abs(tf.subtract(yt,t_test)))
print(f"Test set: Difference between observed and \"true\" data: Mean: {tf.keras.backend.get_value(mean_test_observed)}. Standard Deviation: {tf.keras.backend.get_value(std_test_observed)}\n")

# Calculating the mean and standard deviation in difference between the LS Predicted data and the underlying “true” polynomial curve
# train set
mean_training_ls= tf.math.reduce_mean(tf.abs(tf.subtract(y,preds_train)))
std_training_ls= tf.math.reduce_std(tf.abs(tf.subtract(y,preds_train)))
print(f"Training set: Difference between LS predicted and \"true\" data: Mean: {tf.keras.backend.get_value(mean_training_ls)}. Standard Deviation: {tf.keras.backend.get_value(std_training_ls)}")
# test set
mean_test_ls = tf.math.reduce_mean(tf.abs(tf.subtract(yt,preds_test)))
std_test_ls = tf.math.reduce_std(tf.abs(tf.subtract(yt,preds_test)))
print(f"Test set: Difference between LS predicted and \"true\" data: Mean: {tf.keras.backend.get_value(mean_test_ls)}. Standard Deviation: {tf.keras.backend.get_value(std_test_ls)}\n")





print("\n---------------------------------------------------------------------------------------------------------------------------------------------")
print("\nReport, using printed messages, the mean (and standard deviation) in difference between the “SGD-predicted” values and the underlying “true” polynomial curve.\n")

# Polynomial SGD bullet point
optimal_w_sgd = fit_polynomial_sgd(training_set, 4, 0.000000075295, 1)

preds_train_sgd = tf.Variable(tf.zeros([train_set_size,1]))
preds_test_sgd = tf.Variable(tf.zeros([test_set_size,1]))

# We now predict y^{hat} for train set
for i in range(train_set_size):
    preds_train_sgd[i].assign(polynomial_fun(optimal_w_sgd,train_x[i]))

# We now predict y^{hat} for test set
for i in range(test_set_size):
    preds_test_sgd[i].assign(polynomial_fun(optimal_w_sgd, test_x[i]))

# Calculating the mean and standard deviation in difference between the SGD Predicted data and the underlying “true” polynomial curve
# train set
mean_training_sgd= tf.math.reduce_mean(tf.abs(tf.subtract(y,preds_train_sgd)))
std_training_sgd= tf.math.reduce_std(tf.abs(tf.subtract(y,preds_train_sgd)))

print(f"\nTraining set: Difference between SGD predicted and \"true\" data: Mean: {tf.keras.backend.get_value(mean_training_sgd)}. Standard Deviation: {tf.keras.backend.get_value(std_training_sgd)}")
# test set
mean_test_sgd = tf.math.reduce_mean(tf.abs(tf.subtract(yt,preds_test_sgd)))
std_test_sgd = tf.math.reduce_std(tf.abs(tf.subtract(yt,preds_test_sgd)))
print(f"Test set: Difference between SGD predicted and \"true\" data: Mean: {tf.keras.backend.get_value(mean_test_sgd)}. Standard Deviation: {tf.keras.backend.get_value(std_test_sgd)}\n")







print("\n---------------------------------------------------------------------------------------------------------------------------------------------")
print("\nReport the root-mean-square-errors (RMSEs) in both 𝐰 and 𝑦 using printed messages ----------- Report time spent in fitting/training (in seconds) using printed messages \n")

# Compare the accuracy of your implementation using the two methods with ground-truth on
# test set and report the root-mean-square-errors (RMSEs) in both 𝐰 and 𝑦 using printed
# messages

def compute_rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))


# RMSE for SGD
preds_test_sgd = tf.Variable(tf.zeros([test_set_size,1]))
print("STARTING SGD TRAINING")
start_time = time.time()
optimal_w_sgd = fit_polynomial_sgd(training_set, 4, 0.000000075295, 1)
print("\n Training SGD took %s seconds ---" % (time.time() - start_time))

w = tf.Variable([1., 2, 3, 4])
for i in range(test_set_size):
    preds_test_sgd[i].assign(polynomial_fun(optimal_w_sgd, test_x[i]))
print(f"Root mean square error for Stochastic Gradient Descent between predictions and ground truth: {compute_rmse(yt, preds_test_sgd)}")
print(f"Root mean square error for Stochastic Gradient Descent between weights: {compute_rmse(w, optimal_w_sgd[:-1])}\n")

# RMSE for least squares
preds_test_ls = tf.Variable(tf.zeros([test_set_size,1]))
print("STARTING LEAST SQUARES TRAINING")
start_time = time.time()
optimal_w_ls = fit_polynomial_ls(training_set, 3)
print("Training Least Square took %s seconds ---" % (time.time() - start_time))
for i in range(test_set_size):
    preds_test_ls[i].assign(polynomial_fun(optimal_w_ls, test_x[i]))
print(f"Root mean square error for Least Squares between predictions and ground truth: {compute_rmse(yt, preds_test_ls)}") 
print(f"Root mean square error for Least Squares between weights: {compute_rmse(w, optimal_w_ls)}") 

