#!/usr/bin/env python
# coding: utf-8

# # Yandex Data Science School
# ## Linear Regression & Regularization Exercise.
# 
# 
# ## Outline
# In this exercise you will learn the following topics:
# 
# 1. Refresher on how linear regression is solved in batch and in Gradient Descent 
# 2. Implementation of Ridge Regression
# 3. Comparing Ridge, Lasso and vanila Linear Regression on a dataset

# # Git Exercise
# In this exercise you will also experience working with github.
# 
# You might need to install local python enviroment.
# Installation Instruction for ex2 - working on a local python environment:
# https://docs.google.com/document/d/1G0rBo36ff_9JzKy0EkCalK4m_ThNUuJ2bRz463EHK9I
# 
# ## please add the github link of your work below:
# 

# example: https://github.com/ShannyYekhezkelian/Regression_-_Regularization_Exercise

# ## Refresher on Ordinary Least Square (OLS) aka Linear Regeression
# 
# ### Lecture Note
# 
# In Matrix notation, the matrix $X$ is of dimensions $n \times p$ where each row is an example and each column is a feature dimension. 
# 
# Similarily, $y$ is of dimension $n \times 1$ and $w$ is of dimensions $p \times 1$.
# 
# The model is $\hat{y}=X\cdot w$ where we assume for simplicity that $X$'s first columns equals to 1 (one padding), to account for the bias term.
# 
# Our objective is to optimize the loss $L$ defines as resiudal sum of squares (RSS): 
# 
# $L_{RSS}=\frac{1}{N}\left\Vert Xw-y \right\Vert^2$ (notice that in matrix notation this means summing over all examples, so $L$ is scalar.)
# 
# To find the optimal $w$ one needs to derive the loss with respect to $w$.
# 
# $\frac{\partial{L_{RSS}}}{\partial{w}}=\frac{2}{N}X^T(Xw-y)$ (to see why, read about [matrix derivatives](http://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf) or see class notes )
# 
# Thus, the gardient descent solution is $w'=w-\alpha \frac{2}{N}X^T(Xw-y)$.
# 
# Solving $\frac{\partial{L_{RSS}}}{\partial{w}}=0$ for $w$ one can also get analytical solution:
# 
# $w_{OLS}=(X^TX)^{-1}X^Ty$
# 
# The first term, $(X^TX)^{-1}X^T$ is also called the pseudo inverse of $X$.
# 
# See [lecture note from Stanford](https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf) for more details.
# 

# ## Exercise 1 - Ordinary Least Square
# * Get the boston housing dataset by using the scikit-learn package. hint: [load_boston](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)
# 
# * What is $p$? what is $n$ in the above notation? hint: [shape](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.shape.html)
# 
# * write a model `OrdinaryLinearRegression` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score` (which returns the MSE on a given sample set). Hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.
# 
# * Fit the model. What is the training MSE?
# 
# * Plot a scatter plot where on x-axis plot $Y$ and in the y-axis $\hat{Y}_{OLS}$
# 
# * Split the data to 75% train and 25% test 20 times. What is the average MSE now for train and test? Hint: use [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) or [ShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html).
# 
# * Use a t-test to proove that the MSE for training is significantly smaller than for testing. What is the p-value? Hint: use [scipy.stats.ttest_rel](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_rel.html). 
# 
# * Write a new class `OrdinaryLinearRegressionGradientDescent` which inherits from `OrdinaryLinearRegression` and solves the problem using gradinet descent. The class should get as a parameter the learning rate and number of iteration. Plot the class convergance. What is the effect of learning rate? How would you find number of iteration automatically? Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your features first.
# 
# * The following parameters are optional (not mandatory to use):
#     * early_stop - True / False boolean to indicate to stop running when loss stops decaying and False to continue.
#     * verbose- True/False boolean to turn on / off logging, e.g. print details like iteration number and loss (https://en.wikipedia.org/wiki/Verbose_mode)
#     * track_loss - True / False boolean when to save loss results to present later in learning curve graphs

# In[ ]:


import numpy as np
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


X, y = load_boston(return_X_y=True)
X.shape, y.shape


# In[ ]:


print("number of features (p) is:", X.shape[1])
print("number of samples (n) is:", X.shape[0])


# In[ ]:


# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.

class Ols(object):
    def __init__(self):
        self.w = None

    @staticmethod
    def pad(X):
        X_pad = np.pad(X, ((0, 0), (1, 0)), mode='constant', constant_values=1)
        return X_pad

    def _fit(self, X, Y, ridge=False):
    #remeber pad with 1 before fitting
        self.X = self.pad(X)
        self.y = Y
        if ridge:
            self.w = np.linalg.inv(self.X.T @ self.X + self.ridge_lambda * np.identity(self.X.shape[1],)) @ self.X.T @ self.y
        else:
            self.w = np.linalg.pinv(self.X) @ self.y

    def _predict(self, X):
    #return wx
        return self.pad(X) @ self.w

    def score(self, X, Y):
    #return MSE
        return metrics.mean_squared_error(self._predict(X), Y)


# In[ ]:


lr = Ols()
lr._fit(X,y)
print("training MSE:", lr.score(X, y))
predictions = lr._predict(X)
plt.scatter(y,predictions)
plt.xlabel('Y Train')
plt.ylabel('Predicted Y')


# In[ ]:


mse_train = []
mse_test = []

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = i)
    lr = Ols()
    lr._fit(X_train,y_train)
    mse_train.append(lr.score(X_train, y_train))
    mse_test.append(lr.score(X_test, y_test))

print("Average MSE for train is:",np.mean(mse_train))
print("Average MSE for test is:", np.mean(mse_test))
stats.ttest_rel(np.array(mse_train),np.array(mse_test))


# In[ ]:


# Write a new class OlsGd which solves the problem using gradinet descent. 
# The class should get as a parameter the learning rate and number of iteration. 
# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
# What is the effect of learning rate? 
# How would you find number of iteration automatically? 
# Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your feature first.
class Normalizer():
    def __init__(self):
        self.mu = None
        self.std = None

    def fit(self, X):
        self.mu = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
    
    def predict(self, X):
    #apply normalization - by Zscore
        X_norm = (X - np.expand_dims(self.mu, 0)) / np.expand_dims(self.std, 0)
        return X_norm
    
    def transform(self, X): 
        return (X * self.std + self.mu)

class OlsGd(Ols):
    def __init__(self, learning_rate=0.05, 
               num_iteration=1000, 
               normalize=True,
               early_stop=False,
               verbose=False,
                 track_loss=True):
        super(OlsGd, self).__init__()
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.early_stop = early_stop
        self.normalize = normalize
        self.normalizer_X = Normalizer() 
        self.normalizer_y = Normalizer()   
        self.verbose = verbose
        self.track_loss = track_loss
        self.loss_history = []
        self.iterations = []
    
    def _fit(self, X, Y, reset=True, track_loss=True, ridge=False, ridge_lambda = None):
    #remeber to normalize the data before starting
        #create normalization objects
        if self.normalize:
            self.normalizer_X.fit(X)
            self.normalizer_y.fit(Y)
            X_norm = self.normalizer_X.predict(X)
            y_norm = self.normalizer_y.predict(Y)
            super()._fit(X_norm, y_norm)
        else: 
            super()._fit(X, Y)
        #find best weights using gradiant descent
        self.w = self._step(self.X, self.y, ridge, ridge_lambda)

    def _predict(self, X):
    #remeber to normalize the data before starting
        if self.normalize:
            X_norm = self.normalizer_X.predict(X)
            y_pred = super()._predict(X_norm)
            return self.normalizer_y.transform(y_pred)
        else:
            return super()._predict(X)

    def _step(self, X, Y, ridge, ridge_lambda):
    # use w update for gradient descent
        w = np.zeros((self.X.shape[1], ))
        old_w = w
        for i in range(self.num_iteration):
            if ridge:
                grad = self.X.T @ (self.X @ w - self.y) + ridge_lambda * w
            else:
                grad = self.X.T @ (self.X @ w - self.y) #loss function derivative by w (dL/dw)
            old_w, w = w ,w - self.learning_rate * (2/self.X.shape[0])* grad #update w
            loss = self.compute_loss(w)
            if self.verbose:
                print("Iteration:", i, " loss:", loss)
            if self.track_loss:
                self.loss_history.append(loss)
                self.iterations.append(i)
            if self.early_stop:
                if abs(np.sum(old_w - w)) < 0.001:
                    break
        return w
    
    def compute_loss(self, w): 
        N = len(self.y) 
        l = (1 / N) * np.sum(np.square(self.X @ w - self.y)) 
        return l 
    
    def plot(self):
        plt.plot(self.iterations, self.loss_history, label=f'Alpha: {self.learning_rate}')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Loss") 
        plt.legend()

    


# In[ ]:


#Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
print("For all data set:")
for learning_rate in (0.0001, 0.001, 0.01, 0.1):
    OlsGd_object = OlsGd(learning_rate, verbose = False, early_stop = False)
    OlsGd_object._fit(X, y)
    predicted_y = OlsGd_object._predict(X)
    print("MSE: ", OlsGd_object.score(X, y))
    OlsGd_object.plot()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = i)

print("For the train set:")
for learning_rate in (0.0001, 0.001, 0.01, 0.1):
    OlsGd_object = OlsGd(learning_rate, verbose = False, early_stop = False)
    OlsGd_object._fit(X_train, y_train)
    predicted_y = OlsGd_object._predict(X_train)
    print("MSE: ", OlsGd_object.score(X_train, y_train))
    OlsGd_object.plot()


# In[ ]:


print("For the test set:")
for learning_rate in (0.0001, 0.001, 0.01, 0.1):
    OlsGd_object = OlsGd(learning_rate, verbose = False, early_stop = False)
    OlsGd_object._fit(X_train, y_train)
    predicted_y = OlsGd_object._predict(X_test)
    print("MSE: ", OlsGd_object.score(X_test, y_test))
    OlsGd_object.plot()


# 1. What is the effect of learning rate? 
# * If the learning rate is too low, we might not achive the minimum because we are approaching it very slowly. If it's too high, we might skip it and not find the minimum. 
# 2. How would you find number of iteration automatically?
# * We can set up a delta to calculate the difference in the loss between two iterations and if this delta is low (the loss converges), we will stop.

# ## Exercise 2 - Ridge Linear Regression
# 
# Recall that ridge regression is identical to OLS but with a L2 penalty over the weights:
# 
# $L(y,\hat{y})=\sum_{i=1}^{i=N}{(y^{(i)}-\hat{y}^{(i)})^2} + \lambda \left\Vert w \right\Vert_2^2$
# 
# where $y^{(i)}$ is the **true** value and $\hat{y}^{(i)}$ is the **predicted** value of the $i_{th}$ example, and $N$ is the number of examples
# 
# * Show, by differentiating the above loss, that the analytical solution is $w_{Ridge}=(X^TX+\lambda I)^{-1}X^Ty$
# * Change `OrdinaryLinearRegression` and `OrdinaryLinearRegressionGradientDescent` classes to work also for ridge regression (do not use the random noise analogy but use the analytical derivation). Either add a parameter, or use inheritance.
# * **Bonus: Noise as a regularizer**: Show that OLS (ordinary least square), if one adds multiplicative noise to the features the **average** solution for $W$ is equivalent to Ridge regression. In other words, if $X'= X*G$ where $G$ is an uncorrelated noise with variance $\sigma$ and mean 1, then solving for $X'$ with OLS is like solving Ridge for $X$. What is the interpretation? 
# 
# 

# In[ ]:


class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs,self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda
        self.ridge = True

    def _fit(self, X, Y):
    #Closed form of ridge regression
        super()._fit(X, Y, ridge = self.ridge)
        
    def _predict(self, X):
    #Closed form of ridge regression
        return super()._predict(X)


# In[ ]:


ridge_lambda = 0.5
RidgeLs_object = RidgeLs(ridge_lambda)
RidgeLs_object._fit(X, y)
predicted_y = RidgeLs_object._predict(X)
print("training MSE:", RidgeLs_object.score(X, y))
plt.scatter(y,predicted_y)
plt.xlabel('Y Train')
plt.ylabel('Predicted Y')


# In[ ]:


mse_train = []
mse_test = []

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = i)
    RidgeLs_object = RidgeLs(ridge_lambda)
    RidgeLs_object._fit(X_train,y_train)
    mse_train.append(RidgeLs_object.score(X_train, y_train))
    mse_test.append(RidgeLs_object.score(X_test, y_test))

print("Average MSE for train is:",np.mean(mse_train))
print("Average MSE for test is:", np.mean(mse_test))
stats.ttest_rel(np.array(mse_train),np.array(mse_test))


# In[ ]:


class RidgeLsGd(OlsGd):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLsGd,self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda
        self.ridge = True

    def _fit(self, X, Y):
    #Closed form of ridge regression
        super()._fit(X, Y, ridge = self.ridge, ridge_lambda = self.ridge_lambda)
        
    def _predict(self, X):
    #Closed form of ridge regression
        return super()._predict(X)
    
    def plot(self):
        plt.plot(self.iterations, self.loss_history, label=f'Alpha: {self.learning_rate}')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Loss") 
        plt.legend()


# In[ ]:


ridge_lambda = 0.5
#Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
for learning_rate in (0.0001, 0.001, 0.01, 0.1):
    RidgeLsGd_object = RidgeLsGd(ridge_lambda, learning_rate)
    RidgeLsGd_object._fit(X, y)
    predicted_y = RidgeLsGd_object._predict(X)
    print("MSE:", RidgeLsGd_object.score(X, y))
    RidgeLsGd_object.plot()


# ### Use scikitlearn implementation for OLS, Ridge and Lasso

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)
metrics.mean_squared_error(lr.predict(X_test), y_test)


# In[ ]:


lasso = Lasso()
lasso.fit(X_train, y_train)
metrics.mean_squared_error(lasso.predict(X_test), y_test)


# In[ ]:


ridge = Ridge()
ridge.fit(X_train, y_train)
metrics.mean_squared_error(ridge.predict(X_test), y_test)


# In[ ]:




