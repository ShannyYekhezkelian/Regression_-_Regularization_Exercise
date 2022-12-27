{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "chEqqJbLzFew"
   },
   "source": [
    "# Yandex Data Science School\n",
    "## Linear Regression & Regularization Exercise.\n",
    "\n",
    "\n",
    "## Outline\n",
    "In this exercise you will learn the following topics:\n",
    "\n",
    "1. Refresher on how linear regression is solved in batch and in Gradient Descent \n",
    "2. Implementation of Ridge Regression\n",
    "3. Comparing Ridge, Lasso and vanila Linear Regression on a dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Git Exercise\n",
    "In this exercise you will also experience working with github.\n",
    "\n",
    "You might need to install local python enviroment.\n",
    "Installation Instruction for ex2 - working on a local python environment:\n",
    "https://docs.google.com/document/d/1G0rBo36ff_9JzKy0EkCalK4m_ThNUuJ2bRz463EHK9I\n",
    "\n",
    "## please add the github link of your work below:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "example: https://github.com/ShannyYekhezkelian/Regression_-_Regularization_Exercise"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mR9UFmk2greT"
   },
   "source": [
    "## Refresher on Ordinary Least Square (OLS) aka Linear Regeression\n",
    "\n",
    "### Lecture Note\n",
    "\n",
    "In Matrix notation, the matrix $X$ is of dimensions $n \\times p$ where each row is an example and each column is a feature dimension. \n",
    "\n",
    "Similarily, $y$ is of dimension $n \\times 1$ and $w$ is of dimensions $p \\times 1$.\n",
    "\n",
    "The model is $\\hat{y}=X\\cdot w$ where we assume for simplicity that $X$'s first columns equals to 1 (one padding), to account for the bias term.\n",
    "\n",
    "Our objective is to optimize the loss $L$ defines as resiudal sum of squares (RSS): \n",
    "\n",
    "$L_{RSS}=\\frac{1}{N}\\left\\Vert Xw-y \\right\\Vert^2$ (notice that in matrix notation this means summing over all examples, so $L$ is scalar.)\n",
    "\n",
    "To find the optimal $w$ one needs to derive the loss with respect to $w$.\n",
    "\n",
    "$\\frac{\\partial{L_{RSS}}}{\\partial{w}}=\\frac{2}{N}X^T(Xw-y)$ (to see why, read about [matrix derivatives](http://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf) or see class notes )\n",
    "\n",
    "Thus, the gardient descent solution is $w'=w-\\alpha \\frac{2}{N}X^T(Xw-y)$.\n",
    "\n",
    "Solving $\\frac{\\partial{L_{RSS}}}{\\partial{w}}=0$ for $w$ one can also get analytical solution:\n",
    "\n",
    "$w_{OLS}=(X^TX)^{-1}X^Ty$\n",
    "\n",
    "The first term, $(X^TX)^{-1}X^T$ is also called the pseudo inverse of $X$.\n",
    "\n",
    "See [lecture note from Stanford](https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf) for more details.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JA3MEKz80vdy"
   },
   "source": [
    "## Exercise 1 - Ordinary Least Square\n",
    "* Get the boston housing dataset by using the scikit-learn package. hint: [load_boston](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)\n",
    "\n",
    "* What is $p$? what is $n$ in the above notation? hint: [shape](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.shape.html)\n",
    "\n",
    "* write a model `OrdinaryLinearRegression` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score` (which returns the MSE on a given sample set). Hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.\n",
    "\n",
    "* Fit the model. What is the training MSE?\n",
    "\n",
    "* Plot a scatter plot where on x-axis plot $Y$ and in the y-axis $\\hat{Y}_{OLS}$\n",
    "\n",
    "* Split the data to 75% train and 25% test 20 times. What is the average MSE now for train and test? Hint: use [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) or [ShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html).\n",
    "\n",
    "* Use a t-test to proove that the MSE for training is significantly smaller than for testing. What is the p-value? Hint: use [scipy.stats.ttest_rel](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_rel.html). \n",
    "\n",
    "* Write a new class `OrdinaryLinearRegressionGradientDescent` which inherits from `OrdinaryLinearRegression` and solves the problem using gradinet descent. The class should get as a parameter the learning rate and number of iteration. Plot the class convergance. What is the effect of learning rate? How would you find number of iteration automatically? Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your features first.\n",
    "\n",
    "* The following parameters are optional (not mandatory to use):\n",
    "    * early_stop - True / False boolean to indicate to stop running when loss stops decaying and False to continue.\n",
    "    * verbose- True/False boolean to turn on / off logging, e.g. print details like iteration number and loss (https://en.wikipedia.org/wiki/Verbose_mode)\n",
    "    * track_loss - True / False boolean when to save loss results to present later in learning curve graphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.datasets import load_boston\n",
    "from sklearn import metrics\n",
    "from sklearn.model_selection import train_test_split\n",
    "from scipy import stats\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((506, 13), (506,))"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X, y = load_boston(return_X_y=True)\n",
    "X.shape, y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of features (p) is: 13\n",
      "number of samples (n) is: 506\n"
     ]
    }
   ],
   "source": [
    "print(\"number of features (p) is:\", X.shape[1])\n",
    "print(\"number of samples (n) is:\", X.shape[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "ZuSS8LhcfZdn"
   },
   "outputs": [],
   "source": [
    "# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.\n",
    "\n",
    "class Ols(object):\n",
    "    def __init__(self):\n",
    "        self.w = None\n",
    "\n",
    "    @staticmethod\n",
    "    def pad(X):\n",
    "        X_pad = np.pad(X, ((0, 0), (1, 0)), mode='constant', constant_values=1)\n",
    "        return X_pad\n",
    "\n",
    "    def _fit(self, X, Y, ridge=False):\n",
    "    #remeber pad with 1 before fitting\n",
    "        self.X = self.pad(X)\n",
    "        self.y = Y\n",
    "        if ridge:\n",
    "            self.w = np.linalg.inv(self.X.T @ self.X + self.ridge_lambda * np.identity(self.X.shape[1],)) @ self.X.T @ self.y\n",
    "        else:\n",
    "            self.w = np.linalg.pinv(self.X) @ self.y\n",
    "\n",
    "    def _predict(self, X):\n",
    "    #return wx\n",
    "        return self.pad(X) @ self.w\n",
    "\n",
    "    def score(self, X, Y):\n",
    "    #return MSE\n",
    "        return metrics.mean_squared_error(self._predict(X), Y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training MSE: 21.894831181729202\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Predicted Y')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "lr = Ols()\n",
    "lr._fit(X,y)\n",
    "print(\"training MSE:\", lr.score(X, y))\n",
    "predictions = lr._predict(X)\n",
    "plt.scatter(y,predictions)\n",
    "plt.xlabel('Y Train')\n",
    "plt.ylabel('Predicted Y')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average MSE for train is: 21.23361328923018\n",
      "Average MSE for test is: 25.145172096824204\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Ttest_relResult(statistic=-3.0234950131472793, pvalue=0.006988247970256539)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mse_train = []\n",
    "mse_test = []\n",
    "\n",
    "for i in range(20):\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = i)\n",
    "    lr = Ols()\n",
    "    lr._fit(X_train,y_train)\n",
    "    mse_train.append(lr.score(X_train, y_train))\n",
    "    mse_test.append(lr.score(X_test, y_test))\n",
    "\n",
    "print(\"Average MSE for train is:\",np.mean(mse_train))\n",
    "print(\"Average MSE for test is:\", np.mean(mse_test))\n",
    "stats.ttest_rel(np.array(mse_train),np.array(mse_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Write a new class OlsGd which solves the problem using gradinet descent. \n",
    "# The class should get as a parameter the learning rate and number of iteration. \n",
    "# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.\n",
    "# What is the effect of learning rate? \n",
    "# How would you find number of iteration automatically? \n",
    "# Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your feature first.\n",
    "class Normalizer():\n",
    "    def __init__(self):\n",
    "        self.mu = None\n",
    "        self.std = None\n",
    "\n",
    "    def fit(self, X):\n",
    "        self.mu = np.mean(X, axis = 0)\n",
    "        self.std = np.std(X, axis = 0)\n",
    "    \n",
    "    def predict(self, X):\n",
    "    #apply normalization - by Zscore\n",
    "        X_norm = (X - np.expand_dims(self.mu, 0)) / np.expand_dims(self.std, 0)\n",
    "        return X_norm\n",
    "    \n",
    "    def transform(self, X): \n",
    "        return (X * self.std + self.mu)\n",
    "\n",
    "class OlsGd(Ols):\n",
    "    def __init__(self, learning_rate=0.05, \n",
    "               num_iteration=1000, \n",
    "               normalize=True,\n",
    "               early_stop=False,\n",
    "               verbose=False,\n",
    "                 track_loss=True):\n",
    "        super(OlsGd, self).__init__()\n",
    "        self.learning_rate = learning_rate\n",
    "        self.num_iteration = num_iteration\n",
    "        self.early_stop = early_stop\n",
    "        self.normalize = normalize\n",
    "        self.normalizer_X = Normalizer() \n",
    "        self.normalizer_y = Normalizer()   \n",
    "        self.verbose = verbose\n",
    "        self.track_loss = track_loss\n",
    "        self.loss_history = []\n",
    "        self.iterations = []\n",
    "    \n",
    "    def _fit(self, X, Y, reset=True, track_loss=True, ridge=False, ridge_lambda = None):\n",
    "    #remeber to normalize the data before starting\n",
    "        #create normalization objects\n",
    "        if self.normalize:\n",
    "            self.normalizer_X.fit(X)\n",
    "            self.normalizer_y.fit(Y)\n",
    "            X_norm = self.normalizer_X.predict(X)\n",
    "            y_norm = self.normalizer_y.predict(Y)\n",
    "            super()._fit(X_norm, y_norm)\n",
    "        else: \n",
    "            super()._fit(X, Y)\n",
    "        #find best weights using gradiant descent\n",
    "        self.w = self._step(self.X, self.y, ridge, ridge_lambda)\n",
    "\n",
    "    def _predict(self, X):\n",
    "    #remeber to normalize the data before starting\n",
    "        if self.normalize:\n",
    "            X_norm = self.normalizer_X.predict(X)\n",
    "            y_pred = super()._predict(X_norm)\n",
    "            return self.normalizer_y.transform(y_pred)\n",
    "        else:\n",
    "            return super()._predict(X)\n",
    "\n",
    "    def _step(self, X, Y, ridge, ridge_lambda):\n",
    "    # use w update for gradient descent\n",
    "        w = np.zeros((self.X.shape[1], ))\n",
    "        old_w = w\n",
    "        for i in range(self.num_iteration):\n",
    "            if ridge:\n",
    "                grad = self.X.T @ (self.X @ w - self.y) + ridge_lambda * w\n",
    "            else:\n",
    "                grad = self.X.T @ (self.X @ w - self.y) #loss function derivative by w (dL/dw)\n",
    "            old_w, w = w ,w - self.learning_rate * (2/self.X.shape[0])* grad #update w\n",
    "            loss = self.compute_loss(w)\n",
    "            if self.verbose:\n",
    "                print(\"Iteration:\", i, \" loss:\", loss)\n",
    "            if self.track_loss:\n",
    "                self.loss_history.append(loss)\n",
    "                self.iterations.append(i)\n",
    "            if self.early_stop:\n",
    "                if abs(np.sum(old_w - w)) < 0.001:\n",
    "                    break\n",
    "        return w\n",
    "    \n",
    "    def compute_loss(self, w): \n",
    "        N = len(self.y) \n",
    "        l = (1 / N) * np.sum(np.square(self.X @ w - self.y)) \n",
    "        return l \n",
    "    \n",
    "    def plot(self):\n",
    "        plt.plot(self.iterations, self.loss_history, label=f'Alpha: {self.learning_rate}')\n",
    "        plt.xlabel(\"Number of Iterations\")\n",
    "        plt.ylabel(\"Loss\") \n",
    "        plt.legend()\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "For all data set:\n",
      "MSE:  44.81196897200199\n",
      "MSE:  23.969415656633775\n",
      "MSE:  21.95395438953096\n",
      "MSE:  21.89483118173492\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEGCAYAAABo25JHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAAsTAAALEwEAmpwYAABHi0lEQVR4nO3dd3wUdfrA8c+zJb0XQhoQSOid0EQURBC7giLoeXqKnnd2z/sdXtHTK3rq6dm7oieHXURFUSkqSAsqvYUeOgQIENK/vz9mEpYQkhCy2ST7vF+uuzPz3ZlnsiHPfst8R4wxKKWU8l8OXweglFLKtzQRKKWUn9NEoJRSfk4TgVJK+TlNBEop5edcvg7gVMXFxZk2bdr4OgyllGpSFi9evNcYE1/VtiaXCNq0aUNWVpavw1BKqSZFRDafbJs2DSmllJ/TRKCUUn5OE4FSSvm5JtdHoJTyneLiYnJycigoKPB1KOokgoKCSElJwe121/o9mgiUUrWWk5NDeHg4bdq0QUR8HY6qxBjDvn37yMnJIS0trdbv81rTkIi8LiK7RWT5SbaLiDwtItkislREensrFqVU/SgoKCA2NlaTQCMlIsTGxp5yjc2bfQQTgZHVbD8fyLAfNwMveDEWpVQ90STQuNXl8/FaIjDGfAfkVlPkUuAtY5kPRIlIorfi2bT3CI9+uZrSMp12WymlPPly1FAysNVjOcdedwIRuVlEskQka8+ePXU62Fcrd/L87PXc+c5PFJWU1WkfSqnGYcqUKYgIq1evrli3adMmunbtWu37alPmVLz55ptkZGSQkZHBm2++WWWZ3Nxchg8fTkZGBsOHD2f//v2A1Z5/xx13kJ6eTvfu3fnxxx9r3O+f/vQnUlNTCQsLq7dzgCYyfNQY87IxJtMYkxkfX+UV0jW6+ax23DeyA58t3cGNby4iv6iknqNUSjWUyZMnc+aZZzJ58mSfxZCbm8uDDz7IggULWLhwIQ8++GDFH3lPjzzyCMOGDWPdunUMGzaMRx55BIAvvviCdevWsW7dOl5++WV+85vf1Ljfiy++mIULF9b7ufgyEWwDUj2WU+x13jH/BX79w9k8fnkn5mbv5ZpXF3Agv8hrh1NKecfhw4eZM2cOr732Gu+8806VZSZOnMill17KkCFDyMjI4MEHH6zYVlpayk033USXLl0YMWIER48eBeCVV16hb9++9OjRg9GjR5Ofn19tHNOnT2f48OHExMQQHR3N8OHD+fLLL08o98knn3DdddcBcN111zFlypSK9b/85S8REQYMGMCBAwfYsWNHtfsdMGAAiYn134Luy+GjU4HbROQdoD9w0Bizw2tHc4dA0WGuaO8k7Jo+3DH5J8a8NI+3buhPy8ggrx1WqebqwU9XsHJ7Xr3us3NSBA9c3KXaMp988gkjR46kffv2xMbGsnjxYvr06XNCuYULF7J8+XJCQkLo27cvF154IXFxcaxbt47JkyfzyiuvMGbMGD788EN+8YtfMGrUKG666SYA/vznP/Paa69x++23M3XqVLKysnjooYeO2/+2bdtITT32XTYlJYVt2078Lrtr166KP94tW7Zk165d1b6/tvutT94cPjoZmAd0EJEcEblRRG4RkVvsItOADUA28ArwW2/FAkBkivWct42RXVsy8Ya+bD9QwBUv/sDGvUe8emilVP2ZPHkyY8eOBWDs2LEnbR4aPnw4sbGxBAcHM2rUKObMmQNAWloaPXv2BKBPnz5s2rQJgOXLlzN48GC6devGpEmTWLFiBQCXXHLJCUmgrkSkUY668lqNwBgzrobtBrjVW8c/QXkiOJgDwBnt4ph80wCue2MhV774AxN/1Y+uyZENFo5STV1N39y9ITc3l5kzZ7Js2TJEhNLSUkSExx577ISylf/gli8HBgZWrHM6nRVNQ9dffz1TpkyhR48eTJw4kdmzZ1cbS3Jy8nFlcnJyGDJkyAnlEhIS2LFjB4mJiezYsYMWLVpUvH/r1q3HvT85ObnW+61PTaKzuF5E2AOS7EQA0C0lkvdvGUigy8nYl+fz3dq6jUhSSjWMDz74gGuvvZbNmzezadMmtm7dSlpaGt9///0JZb/++mtyc3M5evQoU6ZMYdCgQdXu+9ChQyQmJlJcXMykSZNqjOW8887jq6++Yv/+/ezfv5+vvvqK884774Ryl1xyScXInzfffJNLL720Yv1bb72FMYb58+cTGRlJYmJirfdbn/wnEQSGQVDUcYkAoF18GB/+5gxSooO5YeIi3svaWvX7lVI+N3nyZC6//PLj1o0ePbrK5qF+/foxevRounfvzujRo8nMzKx233/729/o378/gwYNomPHjhXrp06dyv33339C+ZiYGP7yl7/Qt29f+vbty/33309MTAwA48ePr7hvyoQJ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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.\n",
    "print(\"For all data set:\")\n",
    "for learning_rate in (0.0001, 0.001, 0.01, 0.1):\n",
    "    OlsGd_object = OlsGd(learning_rate, verbose = False, early_stop = False)\n",
    "    OlsGd_object._fit(X, y)\n",
    "    predicted_y = OlsGd_object._predict(X)\n",
    "    print(\"MSE: \", OlsGd_object.score(X, y))\n",
    "    OlsGd_object.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "For the train set:\n",
      "MSE:  37.087868069852526\n",
      "MSE:  20.435870302379207\n",
      "MSE:  18.494643559214698\n",
      "MSE:  18.444113260059986\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEGCAYAAABo25JHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAAsTAAALEwEAmpwYAABHLUlEQVR4nO3dd3xUZdbA8d+Zkt4baUAChBKqEJqAgAoiKqisirq2V2Fd67qru7q6rmWLu7bVta0VK64VcRUBFRQUpErvPdTQEyD9ef+4N2EIIY1MJsmcr59x5t77zJ0zmZAzT71ijEEppZT/cvg6AKWUUr6liUAppfycJgKllPJzmgiUUsrPaSJQSik/5/J1ALUVFxdn0tLSfB2GUko1KQsXLtxrjImv7FiTSwRpaWksWLDA12EopVSTIiJbTnVMm4aUUsrPaSJQSik/p4lAKaX8XJPrI1BK+U5RURHZ2dnk5+f7OhR1CkFBQaSmpuJ2u2v8HE0ESqkay87OJjw8nLS0NETE1+GoCowx7Nu3j+zsbNLT02v8PK81DYnI6yKyR0SWn+K4iMizIrJeRJaKSE9vxaKUqh/5+fnExsZqEmikRITY2Nha19i82UcwARhRxfHzgQz7Nh540YuxKKXqiSaBxq0un4/XEoEx5ntgfxVFRgNvGctcIEpEkrwVz+a9R/jHV6spLdVlt5VSypMvRw2lANs8trPtfScRkfEiskBEFuTk5NTpxaat3MWLMzdw/6RlmgyUauImTZqEiLB69eryfZs3b6ZLly5VPq8mZWrjzTffJCMjg4yMDN58881Ky+zfv59hw4aRkZHBsGHDOHDgAGC1599xxx20a9eObt26sWjRomrPe//999OyZUvCwsLq7T1AExk+aox52RiTZYzJio+vdIZ0tcYNasPtZ7dj4rxt3D9puSYDpZqwiRMnMnDgQCZOnOizGPbv38/DDz/MTz/9xLx583j44YfL/8h7euyxxzjnnHNYt24d55xzDo899hgAU6ZMYd26daxbt46XX36ZX//619We96KLLmLevHn1/l58mQi2Ay09tlPtfV4hKz/jt7v+wO1D0pk4b6smA6WaqLy8PGbPns1rr73G+++/X2mZCRMmMHr0aIYMGUJGRgYPP/xw+bGSkhLGjRtH586dGT58OMeOHQPglVdeoXfv3nTv3p0xY8Zw9OjRKuOYOnUqw4YNIyYmhujoaIYNG8ZXX311UrnPPvuM6667DoDrrruOSZMmle+/9tprERH69evHwYMH2blzZ5Xn7devH0lJ9d+C7svho5OB20TkfaAvcMgYs9Nrr3YkB9k4g99eHE2pOHh+xgZE4C+ju+BwaOeXUrX18OcrWLnjcL2eMzM5gj9f1LnKMp999hkjRoygffv2xMbGsnDhQnr16nVSuXnz5rF8+XJCQkLo3bs3F1xwAXFxcaxbt46JEyfyyiuvcPnll/Pxxx/zy1/+kksvvZRx48YB8MADD/Daa69x++23M3nyZBYsWMAjjzxywvm3b99Oy5bHv8umpqayffvJ32V3795d/sc7MTGR3bt3V/n8mp63PnktEYjIRGAIECci2cCfATeAMeYl4EtgJLAeOArc4K1YAAi3PgjJ3cndw88A4PkZGwBNBko1JRMnTuTOO+8EYOzYsUycOLHSRDBs2DBiY2MBuPTSS5k9ezYXX3wx6enp9OjRA4BevXqxefNmAJYvX84DDzzAwYMHycvL47zzzgNg1KhRjBo1ql5iF5FGOerKa4nAGHNlNccNcKu3Xv8kEXZ1KncnIj25e3gHjIEXZm5AgEc1GShVK9V9c/eG/fv38+2337Js2TJEhJKSEkSExx9//KSyFf/glm0HBgaW73M6neVNQ9dffz2TJk2ie/fuTJgwgZkzZ1YZS0pKygllsrOzGTJkyEnlWrRowc6dO0lKSmLnzp0kJCSUP3/btm0nPD8lJaXG561PTaKzuF6EJ1v3h3cA1i/FPed14JYhbXn3J+0zUKop+Oijj7jmmmvYsmULmzdvZtu2baSnpzNr1qyTyk6fPp39+/dz7NgxJk2axIABA6o8d25uLklJSRQVFfHuu+9WG8t5553HtGnTOHDgAAcOHGDatGnltQhPo0aNKh/58+abbzJ69Ojy/W+99RbGGObOnUtkZCRJSUk1Pm998p9EEJYA4oTc490QZcng1qFtmThvK3d/uITiklIfBqmUqsrEiRO55JJLTtg3ZsyYSkcP9enThzFjxtCtWzfGjBlDVlZWled+9NFH6du3LwMGDKBjx47l+ydPnsyDDz54UvmYmBj+9Kc/0bt3b3r37s2DDz5ITEwMADfddFP5dVPuvfdepk+fTkZGBl9//TX33nsvACNHjqRNmza0a9eOcePG8cILL1R73t///vekpqZy9OhRUlNTeeihh2r4k6uaWC00TUdWVpap84VpnuwEbYbAJSdPYn5+xnoen7qG87sk8szYMwhw+U+OVKqmVq1aRadOnXwdRrUmTJjAggULeO6553wdik9U9jmJyEJjTKXZ0L/+2kUkQe6OSg/dOrQdf7owkynLdzH+7QXkF5U0cHBKKeUb/pUIwpPg8KlHqN44MJ2/X9qV79bmcP0b88grKG7A4JRS9eX666/329pAXfhXIohIPqGPoDJX9mnF05f3YP7mA1zz2k8cOlbUQMEppZRv+F8iKDgMBXlVFrv4jBSev6ony7cf4sqX57Ivr6CBAlRKqYbnX4mgbAhpNbUCgBFdEnnl2iw27s3jspfmsG1/1dPNlVKqqfKvRFA2qexw5R3GFQ3pkMA7N/Zlb14BY178kdW76nc6vVJKNQb+lQhqUSMok5UWw4c3n4kIXPbSHOZtquoSC0qphtDcl6EeMWIEUVFRXHjhhfUWa1X8KxHUskZQpkNiOB//+kziwwO55rWfmL5ytxeCU0rVVHNehhrgnnvu4e23326w9+JfiSAgFAIja1UjKJMaHcJHN59Jx6QIfvX2Av47f6sXAlRKVae5L0MNcM455xAeHl7rn01d+XIZat+ISKp1jaBMTGgA793Ul1+/u4g/fLyMvXmF3DKkbaNcTVApr5tyL+xaVr/nTOwK5z9WZZHmvgy1N643UB3/qhGANamsDjWCMqGBLl69NovRPZJ5fOoaHpi0XNcnUqoBTZw4kbFjxwLHl6GuTNky1MHBweXLUANVLkM9aNAgunbtyrvvvsuKFSsAa3G4ikmgrvxuGepGKyIFNqyuvlwVAlwOnr68B8lRwbw4cwPbDx7juat6Ehbofz9O5ceq+ebuDf6wDLUv+F+NIDIVcndBceFpncbhEP4woiN/v7Qrs9bt5bKX5rDrUH49BamUqow/LEPtC/6XCKJaAgYOZ9fL6a7s04rXr+/Ntv1Hufj5H+r90n1KqeP8YRlqgEGDBnHZZZfxzTffkJqaytSpU2v4E6obry5DLSIjgGcAJ/CqMeaxCsdbA68D8cB+4JfGmCr/Qp/WMtQAG7+Dt0bBtZOhzeC6n6eCVTsP838T5nP4WBHPXd2ToR0S6u3cSjUWugx109BolqEWESfwPHA+kAlcKSKZFYo9AbxljOkGPAL83VvxlIuye+kP1U+NoEynpAg+vWUArWNDuenNBbz705Z6Pb9SSnmLN5uG+gDrjTEbjTGFwPvA6AplMoFv7cczKjle/yJSAYFD26otWluJkUF8cHN/zsqI4/5Pl/Pw5yt0RJFSPqDLUNeONxNBCuD51zbb3udpCXCp/fgSIFxEYiueSETGi8gCEVmQk5NzelG5AiA8EQ7WfyIACAt08cq1WfzfgHTe+GEzN0yYz6GjupS1Uqrx8nVn8d3AYBFZDAwGtgMnXRrMGPOyMSbLGJMVHx9/+q8a2RIOeW9msMvp4MGLMvnnmG7M3biP0c/PZv2eqpe+VkopX/FmItgOtPTYTrX3lTPG7DDGXGqMOQO439530BvB7D6ym8V7FlNqSq1+Ai/VCDxd3rslE8f1I6+gmEue/4EZa/Z4/TWVUqq2vJkI5gMZIpIuIgHAWGCyZwERiRORshjuwxpB5BVfbPqCa6dcS0FJgTWX4PB2KPV++31WWgyf3TaQljEh/N+E+bz8/Qa8OVJLKaVqy2uJwBhTDNwGTAVWAR8YY1aIyCMiMsouNgRYIyJrgRbAX70Vj1OcAJSUllhNQyWFkNcwq4imRAXz0a/7M7JLEn/7cjW/+3AJ+UUntYAppWqoOSxDvXr1avr3709gYCBPPPFEvcVUF17tIzDGfGmMaW+MaWuM+au970FjzGT78UfGmAy7zE3GGK9dE7I8EZgSiGpl7fTCyKFTCQlw8dxVZ3DXue35ZNF2fvHSj3rVM6XqqDksQx0TE8Ozzz7L3Xff3dChn8TXncUNxumwEkFxabFVI4AGTQRgrXVy57kZvHZdFlv2HeWi52bz3drTHAWllJ9pLstQJyQk0Lt3b9xud11+DPXKb1ZJK6sRlHcWQ4N0GFfmnE4t+Py2gdz8zkKuf2Med53bntuGtsPhaHyrEip1Kv+Y9w9W7z+9BRwr6hjTkT/0+UOVZZrLMtSNid/UCFwOK+eVmBIIDIegqAavEXhKiwvl01sGcHGPFJ6avpZxby3g0DGdb6BUdXQZ6vrnNzUChz04qbi02NoR1QoO+vYqY8EBTp66vDtntIrikc9XMuq52bz0y150SorwaVxK1UR139y9oTktQ92Y+E2N4ISmIYDoNNi/yXcB2USEa/un8d9f9SO/qIRLXviBD+Zv0yGmSlWiOS1D3Zj4TSIoaxoqNnaNICYdDm6B0sYxjLNX6xj+d/sgerWO5vcfL+Wu//5MXkGxr8NSqlFpTstQ79q1i9TUVJ566in+8pe/kJqayuHDvlnG3qvLUHtDXZehnrp5Knd/dzefjPqEjOgMWPAG/O838JvlxzuPG4GSUsMLM9bz9NdraR0bynNXnUHn5Ehfh6UUoMtQNxWNZhnqxsYlHp3FADFtrPsDvm8e8uR0CLefk8HEcf04WljMJS/8yNtzNmtTkVLKa/wmEZTNIzieCNKt+/0bfRRR1fq2ieXLOwZxZttY/vTZCm55d5GOKlKqhnQZ6trxn0TgucQEWBexd7gbRYfxqcSGBfL6db257/yOTF+5mwuencXirSfPXFRKqdPhf4mgrEbgcEJ060bXNFSRwyH8anBbPri5P8bAL16awzNfr9ML3iil6o3/JAJHhRoBQHR6o64ReOrZKpopvxnEqO7JPP31Wi7/zxy27Dvi67CUUs2A/ySCijUCsDqM92+CJtIRGxHk5ukrevDslWewbk8eI5+ZxYcLdM6BUur0+E8iqKxGEJMOhblwdJ+PoqqbUd2T+eo3Z9ElJZJ7PlrKre8t4sCRQl+HpVSDaUrLUH/44Yd07twZh8NBXYa+NwT/SQR2jaB8QhlYTUPQZJqHPKVEBfPeuH7ca3ckj3jme2at05VMlX9oSstQd+nShU8++YSzzjrLB1HWjP8kAkeFJSag0Q8hrY7TIdw8uC2f3jKAsEAX17w2jz9+ukxnJKtmraktQ92pUyc6dOhwGu/Y+/xm0bnyCWWeTUNRrQFpsomgTJeUSL64YxBPTlvDq7M38d2aHB7/RTfObBfn69BUM7brb3+jYFX9LkMd2KkjiX/8Y5Vlmtoy1E2BV2sEIjJCRNaIyHoRubeS461EZIaILBaRpSIy0luxlK8+6tk05A6ylpfYt95bL9tggtxO7r8gkw9/1Z8Al4OrXv2JP01azhGtHahmpikvQ91Yea1GICJO4HlgGJANzBeRycaYlR7FHsC6lvGLIpIJfAmkeSOeSpuGAOLaw9613nhJn8hKi+HLOwbxxLQ1vP7DJmau3cM/x3Snf9tYX4emmpnqvrl7Q1Nchrop8GaNoA+w3hiz0RhTCLwPVFx/1QBli+9HAju8FUxZ01D59QjKxLW3agSlzWeCVnCAkz9dmMkHv+qPU4QrX5nLg59p7UA1fU1xGeqmwJuJIAXwvARYtr3P00PAL0UkG6s2cLu3gilrGjphHgFAXAYUHYXDTbNtryq902KYcudZ3DAgjbfnbmH4098zY/UeX4elVJ01xWWoP/30U1JTU5kzZw4XXHBBo0wWXluGWkR+AYwwxtxkb18D9DXG3OZR5rd2DE+KSH/gNaCLMSe234jIeGA8QKtWrXpt2bKl1vHsOrKLYR8N46H+DzGm/ZjjBzb/ABNGwi8/gXbn1Pq8TcWCzfu595NlrN+Tx0Xdk3nwwkziwwOrf6JSHnQZ6qahMS1DvR3wXOg/1d7n6UbgAwBjzBwgCDhpqIsx5mVjTJYxJis+Pr5OwVQ6sxispiGAvevqdN6mIisthi/uGMhd57Zn6vJdnPPkTP47f6vOSlZKeTURzAcyRCRdRAKAscDkCmW2AucAiEgnrETglVlRJy1DXSY0zrqQfTPqMD6VQJeTO8/N4Ms7B9ExMYI/fLyMsS/PZWNOnq9DU6pe6TLUteO1RGCMKQZuA6YCq7BGB60QkUdEZJRd7HfAOBFZAkwErjde+op60jLUZUSa3cih6rRLCOP98f147NKurNp5mBHPzOLZb9ZRUNw4LtupGjetRTZudfl8vDqhzBjzJVYnsOe+Bz0erwSq7sqvJ6dsGgIrEaz/uiHCaDQcDmFsn1ac3SmBhyev5Knpa/l08XYeGtWZwe3r1vymmr+goCD27dtHbGzsScMzle8ZY9i3bx9BQUG1ep7fzCw+ZdMQWCOHfn4H8g9BkH9dHzghPIjnr+7J5WtzeGjyCq57fR4jOifyp4sySYkK9nV4qpFJTU0lOzubnBxd16qxCgoKIjU1tVbP8ZtEUOkSE2U8O4xTqx5i1lwNbh/PV78ZxKuzNvHvb9cx88k93H52BjcNSifQ5fR1eKqRcLvdpKen+zoMVc/8ZtG5SpeYKBNvLwiVU7/rpjQ1gS4ntw5tx9e/HcyQ9gk8PnUNI/41i+/X6rc/pZozv0sEJy0xARCdBq5g2L3y5GN+KDU6hJeu6cWEG3pjjOHa1+dx89sL2bqv6tUYlVJNk98kAhHBJa7Km4YcTkjoCHtWNHxgjdiQDglMvess7h7enu/W5nDuU9/x2JTV5OYX+To0pVQ98ptEAFatoNKmIYAWnWG3JoKKAl1Objs7gxl3D+HC7km89N0Ghj4xk/fnbaWkVIcRKtUc+FUicDqclJ5qcbmEznAkB/J0LZ7KJEYG8dTlPfjs1gG0jg3l3k+WceG/Z/Pjhr2+Dk0pdZr8KhG4xFX58FGwagSgtYJqdG8ZxUc39+e5q87g8LEirnrlJ8a9tYBNe4/4OjSlVB35VSJwOpwnL0NdRhNBjYkIF3ZL5pvfDeae8zrw4/q9DHvqOx78bDk5uQW+Dk8pVUt+lQgc4jh1jSA0DsJawB4dOVRTQW5ruOmMe4Ywtk9L3v1pK4Mfn8FT09fqdZOVakL8KhG4xFX58NEyCZmwe3nDBdRMJIQH8ZeLuzL9rrMY2iGBZ79Zx+B/zmDCD5soLG4+F/xRqrnyq0RQZdMQWM1De1ZDiX6brYs28WE8f3VPJt06gPYtwnno85Wc+9R3fPbzdkp1hJFSjZZfJYIqm4bASgQlBbB/Q8MF1Qz1aBnFe+P68ub/9SE00MWd7//Mhf+ezTerduvKlUo1Qn6VCFyOU0woK5PU3brfuaRhAmrGRITB7eP54vaB/OuKHuQVFHPjmwu4+PkfmLlmjyYEpRoRv0oEbof71BPKAOI6WEtN7FjccEE1cw6HcPEZKXzzu8H8Y0xX9uYVcv0b8/nFS3P4Yf1eTQhKNQJ+lwiKSqpYHsHpgsSumgi8wO10cEXvVsy4ewh/vaQLOw4e4+pXf+KKl+cyd+M+X4enlF/zq0TgcrgoKq1mnZzkM2DnUqiqCUnVWYDLwdV9WzPzniE8Mrozm/ceYezLc7nqlbnM27Tf1+Ep5Zf8KhG4He4aJIIeUHSk2V/M3tcCXU6u7Z/G978fyoMXZrJ2dx6X/2cOl730o/YhKNXAvJoIRGSEiKwRkfUicm8lx58WkZ/t21oROejNeNwOd9XDR8GqEYA2DzWQILeT/xuYzqzfD+WhizLJPnCM69+Yz6jnfuCr5Tt12KlSDcBriUBEnMDzwPlAJnCliGR6ljHG3GWM6WGM6QH8G/jEW/EAuJw1aBqKaw/uENj5szdDURUEBzi5fkA6390zlH+O6UZeQTE3v7OI4f/6nk8WZVNUohPTlPIWb9YI+gDrjTEbjTGFwPvA6CrKXwlM9GI8NWsacjghsZvWCHwkwOXg8t4t+fq3g3n2yjNwOYTffrCEoU/M5J25W8gv0r4bpeqbNxNBCrDNYzvb3ncSEWkNpAPfnuL4eBFZICILTuei2dWOGipT1mGsM4x9xukQRnVPZsqdg3j12iziwgJ5YNJyBv7jW579Zh37jxT6OkSlmo3G0lk8FvjImMqn/RpjXjbGZBljsuLj4+v8Ii6Hq+p5BGVSs6D4mK471AiICOdmtuDTW87kvXF96ZoSyVPT13LmY9/wp0nL2azLXyt12lxePPd2oKXHdqq9rzJjgVu9GAtQixpBy77W/bafrFFEyudEhDPbxnFm2zjW7s7l1Vkb+e/8bbzz0xaGZ7Zg/Flt6NU6xtdhKtUkebNGMB/IEJF0EQnA+mM/uWIhEekIRANzvBgLUMM+AoDIVAhPthKBanTatwjnn7/ozux7h3LrkHbM3bifMS/O4ZIXfmDKsp0Ua8eyUrXitURgjCkGbgOmAquAD4wxK0TkEREZ5VF0LPC+aYCB4zWaUAYgAq36wrZ53g5JnYaE8CDuPq8Dc+47m0dGd2ZfXiG/fncRgx+fyYszN3BA+xGUqhFvNg1hjPkS+LLCvgcrbD/kzRg81bhGAFbz0IpP4dB2iKy0j1s1EiEBLq7tn8bVfVszfeVu3vxxM//4ajX/+noto3skc92ZaXROjvR1mEo1Wl5NBI2N21mDCWVlPPsJIi/1XlCq3jgdwoguiYzoksiaXbm8OWczny7azgcLsumdFs11Z6ZxXudE3M7GMkZCqcbBr/5F1KpGkNjVWolUm4eapA6J4fztkq7Mve8cHrigE7sPF3Dbe4vLh5/uyc33dYhKNRr+VSNwuCk1pZSUluB0OKsu7HRDSi/Y6vU+bOVFkSFubhrUhhsGpDNzzR7enLOFp6av5dlv1jEsswVX9mnFwHZxOBzi61CV8hm/SgQuh/V2i0qLqk8EAGkD4PvH4dhBCI7yamzKu5wO4ZxOLTinUws25uTx/vxtfLQwmynLd9EyJpixvVtxWVYqCeFBvg5VqQbnd01DQM2bh9LPAlMKW370YlSqobWJD+OPIzsx576zefbKM0iJCubxqWs48+/fcsu7C5m1LkcXu1N+xa9qBGWJoMYdxqm9wRUEm2dBx5FejEz5QqDLyajuyYzqnsyGnDzen7eVjxZm8+WyXbSKCeGK3i0Z0zOVxEitJajmrUY1AhEJFRGH/bi9iIwSEbd3Q6t/bmctawSuQGjVDzZ978WoVGPQNj6M+y/IZM595/DM2B4kRQZZtYTHvuHa1+fx+ZIduuCdarZqWiP4HhgkItHANKxZw1cAV3srMG+oddMQWM1D3zwCR/ZCaJyXIlONRZDbyegeKYzukcLmvUf4eFE2Hy/M5vaJi4kIcnFR92Quy2pJ99RIRLSDWTUPNU0EYow5KiI3Ai8YY/4pIj97MS6vKO8srsl6Q2XSB1v3m2dB50u8EJVqrNLiQvnd8A7cdW575mzcx4cLrA7md3/aSruEMH7RK5VLz0ghIUKbjlTTVuNEICL9sWoAN9r7ajDspnGpU40gqQcEhMPG7zQR+CmHQxjQLo4B7eJ4JL+IL5fu5MOF2Tw2ZTX//Go1gzLiGd0jmeGdEwkL9KtuN9VM1PS39jfAfcCn9npBbYAZXovKS2rdWQzgdFnNQ+u/AWOsdYiU34oIcjO2TyvG9mnFxpw8Pl6UzaTFO/jtB0sIci/j3E4tuLhHCme1jyfA5VeD8lQTVqNEYIz5DvgOwO403muMucObgXlDnWoEABnDYM0XkLMGEjp6ITLVFLWJD+Oe8zryu2EdWLT1AJ/9vIP/Ld3B/5buJDLYzciuSVzcI5neaTE6YU01ajVKBCLyHnAzUILVURwhIs8YYx73ZnD1zXNCWa1kDLfu103TRKBO4nAIWWkxZKXF8OBFmcxet5dJP29n0uLtTJy3leTIIC7qkcxF3ZLpnByhncyq0alp01CmMeawiFwNTAHuBRYCTSoRlNUICktquTxxZAq06GIlggFNriKkGpDb6WBoxwSGdkzgaGEx01fu5rOfd/DarE3857uNtI4N4fwuSVzQNYkuKZoUVONQ00TgtucNXAw8Z4wpEpEmN/UyyGWN7qh1IgCreejHf0P+YQiKqOfIVHMUEuAqH4q6/0gh01bs4otlO3ll1kZe+m4DLWOCGdkliZFdk+imw1GVD9U0EfwH2AwsAb63LzZ/2FtBeUuAMwCA/JI6rDyZMRxmPw0bZ0Dm6HqOTDV3MaEB5Z3MB44UMn3lbr5YtpPXZm/iP99vJCUqmJFdExnZNYkeLaM0KagGVdPO4meBZz12bRGRod4JyXuCnKdRI0jtA0FRsPpLTQTqtESHBnB575Zc3rslh44WMW3lLr5ctpMJP27mlVmbSI4MYnjnRIZltqBPeoxeP0F5XU07iyOBPwNn2bu+Ax4BDlXzvBHAM1hzDl41xjxWSZnLgYcAAywxxlxV0+BrK9AZCNSxRuB0QYeRsPoLKC4EV0A9R6f8UWSIm8uyWnJZVksOHSvi65W7mbJ8JxPnbWXCj5uJCHJxdscEhmUmMrhDvM5TUF5R09+q14HlwOX29jXAG8ApL90lIk7geWAYkA3MF5HJxpiVHmUysOYnDDDGHBCRhNq/hZor6yMoKC6o2wkyR8OS96y1hzLOrcfIlILIYDdjeqUyplcqRwuLmbVuL9NW7Obb1buZ9PMOApwOzmwXy7DMFgzr1EJnNKt6U9NE0NYYM8Zj++EaLDHRB1hvjNkIICLvA6OBlR5lxgHPG2MOABhj9tQwnjo5rT4CgLZDrVnGqz7TRKC8KiTAxXmdEzmvcyLFJaUs3HKA6St3M23lbu7/dDn3f7qc7i2jGJ7ZgrM7JtAxMVz7FVSd1TQRHBORgcaY2QAiMgA4Vs1zUoBtHtvZQN8KZdrb5/sBq/noIWPMVxVPJCLjgfEArVq1qmHIJytrGqpTHwFYq5G2Pw9W/Q8ueNpqLlLKy1xOB33bxNK3TSz3X9CJtbvzmL5yF9NX7ubxqWt4fOoaEiOCGNoxniEdEhjQLk6bkFSt1PS35WbgLbuvAOAAcF09vX4GMARIxRqR1NUYc9CzkDHmZeBlgKysrDoPW3WIgwBHQN1rBACZo2D5R7BlNrQZUvfzKFUHIkKHxHA6JIZz29kZ7D6cz3drcpixZg+fL9nJxHnbcDuFPukxDO2QwJAOCbSND9XagqpSTUcNLQG6i0iEvX1YRH4DLK3iaduBlh7bqfY+T9nAT8aYImCTiKzFSgzzaxZ+7QW6AuveRwDQbpjVPLT0A00EyudaRASVj0AqLLaakGau2cOMNXv4yxer+MsXq0iNDmZohwSGdoynX5tYQgK0tqBOVKvfCGOM59yB3wL/qqL4fCBDRNKxEsBYoOKIoEnAlcAbIhKH1VS0sTYx1VagM5CCktNIBAEh0Hk0rJgEI5+wtpVqBAJcDvq3jaV/21juG9mJ7ANHmbkmh5lr9vDRwmzenruFAKeDnq2jGJQRz6CMODonR+LUdZD83ul8Najyt8cYUywitwFTsdr/X7dXLn0EWGCMmWwfGy4iK7HWMbrHGLPvNGKq1mknAoDuV8Lid6yhpN0uq5/AlKpnqdEh/LJfa37ZrzX5RSXM37yfWev2Mmvd3vK+hagQNwPaWktsD8qIo2WMfrHxR6eTCKptqzfGfAl8WWHfgx6PDVbN4renEUetBDmDTj8RtDoTIlvBkomaCFSTEOR22rWAeABycgv4cYOVFGav28sXy3YCkBYbwsCMOAa2i6d/21gig5vcFWlVHVSZCEQkl8r/4AsQ7JWIvCzQFUh+8Wl0FgM4HND9Cpj1JBzeCRFJ9ROcUg0kPjywfB0kYwwbcvLKk8Kni7bzztytOAS6pkTSr00s/drEkpUWTXiQJobmqMpEYIwJb6hAGkq9NA0BdBsL3z8OS9+HgXed/vmU8hERoV1COO0SwrlhQDpFJaUs3nqQ2ev3MnfDPl7/wVoPyekQuqRE0q9NDP3axNI7LUaHqTYTfvcpBjoDOVZc3RSIGohrB60HwoLX4cw7wNHkrtypVKXcTgd90mPokx4Dw+BYYQmLtx5g7sZ9zNm4j9dnW0tqa2JoPvzuUwtyBnGw4GD9nKzPTfDh9bD+a2uimVLNUHCAkzPbxXFmuzjASgyL7MQwt2JiSI4gKy2G3mnR9GodQ3x4oI+jVzXhd4mgXvoIynS8EMISYf6rmgiU3wgOcDKgnTXSCOBoYTGLthxk7sZ9/LRpH2/P3cJrszcBVudzVloMWa2jyUqL0cltjZT/JQJn4OnNLPbkdEOv6+G7f8D+TRCTXj/nVaoJCQlwWSONMqzEUFBcwvLth1mweT/zNx/gm1W7+WhhNgDRIW56tbZqDFlp0XRJiSTQpc2qvuZ3iSDEFcLRoqP1d8Je11mdxvNfhfP+Wn/nVaqJCnQ56dU6ml6to/nVYOxRSUdYuMVKDAu3HODrVbsBaxJc99RIerSM4oxW0fRoGUVSZJDWGhqY3yWCUHcoR4uOYoypn1+2iGTociksnABn3Q3B0ad/TqWaEWtUUhjtEsK4ore1aGRObgELtxxgweb9LNx6gDd/3MIrs6zmpITwQHq0jKJHqyjOaBlNt9RIQrUT2qv87qcb6g6l2BRTVFpUviz1aRvwG1j2Icx7FQbfUz/nVKoZiw8PZESXREZ0SQSs5qRVO3P5eesBft52kJ+3HWTaSqvW4BBo3yLcSg52zaFdQpgujVGP/C4RhLitKfRHio7UXyJI7AIZ58FPL0L/W3X9IaVqKdDlLP9DX+bAkUJ+zj7I4q1WYpiyfBfvz7dWtg8NcNIlJZJuqZF0SYmka0okabGhODQ51In/JQLX8UQQHVSPzTgD74I3RsDit6Hvr+rvvEr5qejQAGvV1A7WhQuNMWzae6S8xrAk+xBvztlCYXEpAOGBLjKTI+iaEklXO0Gka3KoEb9LBKHuUMBKBPWqdX9o1R9mPw09rwV3k1yBQ6lGS0RoEx9Gm/gwLu2ZCkBRSSnrduexfPshltm3t+duocBODmEeyaGbJodT8ttEUC+ziys6+wGYcAHMewUG3FH/51dKncDtdJCZHEFmcgSX97Yuf1JUUsr6PXksyz6eHN7xSA6hAU46JkXQKSmczKRIOiVZF/rx5+s0+N079+wjqHdpA6HduTD7KWtYaVBk9c9RStUrt9NBp6QIOiUdTw7FJaWs25PHsu2HWLH9EKt25vLZ4h28M3crACKQHhtKp+QIMu0k0SkpgsQI/xjK6n+JwOXFRABwzoPwn7Pgx39bNQSllM+5PJIDWVZyMMaQfeAYK3ceZpV9W5p9kC+W7ix/XnSIu/x5newE0TY+jCB385oE53eJoKxp6GhxPU4q85TUHTpfCj8+Bz2vg6iW1T9HKdXgRISWMSG0jAnhvM6J5ftz84tYvSuXVTsPs3KHlSA8m5YcAmlxobRPCKd9YjjtW4TRoUU4aXGhuJ0OX72d0+J3icCrTUNlhj0Ma6bAtAfg8je99zpKqXoXHuSmd1oMvdNiyveVlFojllbtPMy63bms3Z3H2t25TFu5i1L7ii1up9AmLsxKDgn2fYtwWsWENPo5D36XCLw2ashTVCsY9FuY8VfY+B20Gey911JKeZ3TcXx2tKf8ohI25FhJYe3uPNbuymXx1gN8vmRHeZlAl4OMFmG0Twgno0U4beNDaZsQRuuYEFyNpAbh1UQgIiOAZ7CuWfyqMeaxCsevBx7Hurg9wHPGmFe9GVOgM5BAZyC5hbnefBnrGgWL34Epv4dfzQJXPU1eU0o1GkFuJ52TI+mcfOLAkCMFxazbYyeIXbms2Z3LDxv28sni7eVl3E6hdWyolRjirSTTNj6MNvGhDX4lOK8lAhFxAs8Dw4BsYL6ITDbGrKxQ9L/GmNu8FUdlIgMiOVx42Lsv4g6CkU/Ae5fBrCdg6B+9+3pKqUYjNNB10kxpgMP5RWzMOcKGPXmsz8mz7vfk8c2qPRSXHr8qcIuIwPLEUHZrlxBGi4hAr4xi8maNoA+w3hizEUBE3gdGAxUTQYOLCIzgUMEh779Q++HWJS1nPQkdL7A6kpVSfisiyF1pgigqKWXr/qOs35PHhpw8Nuw5woacPD5dtJ3cguLycn+6MJMbB9b/cvfeTAQpwDaP7WygbyXlxojIWcBa4C5jzLaKBURkPDAeoFWrVqcdWERAhPdrBGVG/B02zoBJt8K4b7WJSCl1ErfTUf7N35MxhpzcAqv2kHOEvukxpzjD6fF1T8XnQJoxphswHah0iI0x5mVjTJYxJis+Pv60XzQyMLJhagQAITFw4b9g9zKY+beGeU2lVLMgIiREBHFm2ziu6dea9i3CvfI63kwE2wHPQfSpHO8UBsAYs88YU2Bvvgr08mI85Ro0EQB0HGmtPzT7aev6xkop1Yh4MxHMBzJEJF1EAoCxwGTPAiKS5LE5CljlxXjKNWjTUJkR/4CETPjkV3B4Z/XllVKqgXgtERhjioHbgKlYf+A/MMasEJFHRGSUXewOEVkhIkuAO4DrvRWPp8jASI4VH6OopKghXs4SEAKXTYCio/DxjVBc2HCvrZRSVfBqH4Ex5ktjTHtjTFtjzF/tfQ8aYybbj+8zxnQ2xnQ3xgw1xqz2VixH5s5l99//jiksJDLAGvN7sOCgt16ucvEd4KJnYcsP8OXdYEz1z1FKKS/zdWdxg8lfsYL9b76FKSoiLjgOgL3H9jZ8IN0ug0G/g0VvwtwXG/71lVKqAr9JBDit1QJNaSlxIVYiyDmW45tYhj4AnS6CaffDmq98E4NSStn8JhGIw04ExcW+rREAOBxwyX8gsRt8eD1s+dE3cSilFH6UCChb3Km01PeJACAgFH75sbVM9buXw47FvotFKeXX/CYRSFnTUEkJgc5AIgIiyDnqo6ahMqFxcM0kCI6Gty+F3St8G49Syi/5TSLAcbxGABAfHO/bGkGZyBS4dhK4Aq3rHW9f5OuIlFJ+xm8SgTjtZZVKSgCIC47zXWdxRbFt4YYpEBgOb47SPgOlVIPym0RQ1kdgyhJBSFzjqBGUiUmHG76C8ESrmUhHEymlGojfJALPPgKwmoZyjuZgGtOkrsgUq2YQ3x7evxLmvqSTzpRSXuc3iaBiH0FiaCKFpYXsy9/nw6AqERZvJYMOI+GrP1gzkEuKq3+eUkrVkd8kgoo1gpbh1sKo2bnZPovplAJC4fK3rctdzn8V3hoNubt9HZVSqpnym0RQNrO4rEaQGpYKwLbck66D0zg4HDD8UWvi2faF8J9BsGmWr6NSSjVDfpMIymsExVaNICU8BYDsvEZYI/DUfax1ZbPACHhrFHz/uDYVKaXqld8kguN9BFYiCHQGkhCS0DibhipqkQnjZ0DnS+Dbv8Ab58O+Db6OSinVTPhNIqjYRwBWP0GTSARgzTEY8xpc+irsXQMvDYR5r+ioIqXUafObRFBx1BBY/QRbc7f6KKA6ELGWsb5lLrTqZ40omnAB5KzxdWRKqSbMbxJBZTWCtlFt2Xtsb8Nev7g+RCTDLz+xLnKzewW8OAC+fhgKj/o6MqVUE+TVRCAiI0RkjYisF5F7qyg3RkSMiGR5LZayUUMeiSAjOgOAdQfWeetlvUcEel0Hty+ErpfB7Kfghb6wcrI2FymlasVriUBEnMDzwPlAJnCliGRWUi4cuBP4yVuxAMcvTFNyvGmoXVQ7ANYdbIKJoExoHFzyIlz/BbhD4YNr4PURkL3A15EppZoIb9YI+gDrjTEbjTGFwPvA6ErKPQr8A8j3YixIhVFDAC1CWhAeEM76A+u9+dINI20g3DwbLvwX7N8Ir54DH95gPVZKqSp4MxGkAJ6ztbLtfeVEpCfQ0hjzRVUnEpHxIrJARBbk5NRxxdBK+ghEhIyojKZdI/DkdEHWDXDHIhj8B1gzBf6dBZ/dCvs3+To6pVQj5bPOYhFxAE8Bv6uurDHmZWNMljEmKz4+vm4vWMmoIYDM2ExW719NcWkzmqQVGA5D/wh3/gx9xsPSD+G5LPjsNk0ISqmTeDMRbAdaemyn2vvKhANdgJkishnoB0z2VoexuKzrEXjWCAC6xXfjWPGxptlhXJ3wRDj/MbhzCfS+CZZ+AP/uBR/fBDt+9nV0SqlGwpuJYD6QISLpIhIAjAUmlx00xhwyxsQZY9KMMWnAXGCUMcYrvZzlfQQVEkH3+O4ALMlZ4o2XbRwikuD8f1gJod+vrWsdvDwYJlwIa6eeVEtSSvkXryUCY0wxcBswFVgFfGCMWSEij4jIKG+97inZVygrW2uoTFJoEnHBcSzNWdrgITW4iCQ476/w2xUw7FGrI/m9y+GFftYs5fzDvo5QKeUDLm+e3BjzJfBlhX0PnqLsEG/GIgFu63WKik7cL0L3+O7Nu0ZQUVAkDLjDqh2s+BTmPGfNUp7+Z+j6C6vDOfkMX0eplGog/jOz2B0AgCksPOlY9/jubM3d2rguXdkQnG7odjmM/85a4bTLpVY/wstDrNvCCZDfxGZdK6VqzX8SQVmNoJJE0C+pHwA/7vDTi8aLQEovGP0c/G41nP84FOXD53fCE+3hw+utfoWSompPpZRqevwmETgC7BpB0cmJoENMB2KCYvw3EXgKjoK+4+GWOVYtoee1sPE7mHgFPNkRpvzBulCOLmOhVLPh1T6CxkQCTt005BAH/ZP7M2fHHEpNKQ7xm/x4amW1hJReMPyvsP5rWPo+LHgdfnoJolpB5mjIvNgqI+LriJVSdeQ3iQCXC0QorSQRAAxIHsAXG79g1f5VdI7t3MDBNXKuAOg40rodOwCr/gerJsPcl+DHf0NEKmSOshJDam9wOH0dsVKqFvwmEYgI4nZXWiMAGJgyEKc4mb55uiaCqgRHQ89rrNuxg7D2K1j5Gcx/Dea+ACFxkDEMMoZD27OtpialVKPmN4kArOYhU1h5h2d0UDT9kvrx1eavuLPnnYg2dVQvOMq6pnL3sdYchHXTrAlqa7+CJRNBnNCqP7QfDhnnQXwHbUJSqhHyw0RQeY0A4Ly083jwxwdZsW8FXeK6NGBkzUBQhDUHoesvrBVesxdYCWHdNJj+oHWLSIH0wdBmsHUfkeTrqJVSaCI4wdmtzubRuY/y+YbPNRGcDocTWvW1buf+GQ5lw7rpsHGmXVt4zyoX1/54YkgbaDU7KaUanH8lgsCqE0FkYCTntj6Xzzd8zp097yTEHdKA0TVjkanWbOWsG6x1jXYvs4akbvoOfn4X5r8CCCRkWtdiLrtFttSmJKUagF8lAkdwCKX5VV//5ooOVzBl0xS+2vwVl2Zc2kCR+RGHA5K6W7cBd0BxIWxfAJtmwba51szmBa9ZZcOT7aTQH1r2gRadrdnQSql65WeJIJjSo0eqLNMzoScZ0Rm8vfJtLm53sc4p8DZXALQ+07qB1b+wewVs+wm2zoGtc2HFJ9YxZyAkdrXWQUrpCck9IS5Dh6sqdZr8KxGEhFCSl1tlGRHhxi43cu+se/l267ec2/rcBopOAdYf9aRu1q3POGvfwW2QPQ+2L4Idi+Hn9+zmJCAgDJJ6QMoZkNjdqjXEZWjNQala8LtEULxnd7XlRqSN4KUlL/HikhcZ2nIoTv3G6VtRLa1blzHWdmkJ7F0HOxbZyWER/PQfKLH7f5wB1lDVFl2txJDYBVp0gdA4370HpRoxv0sEpUeOVlvO6XBya49buef7e/h43cdc3uHyBohO1ZjDCQkdrVuPq6x9JUVWcti93LrtWg4bvj0+QgkgLNF6TlwHq9YQ38EauRTWQjullV/zr0QQGkLp0eoTAVhzCj5c+yHPLHqGYa2HER2kQxsbNacbWmRaNzwS95G9dnJYYSWHnNVW01KhRxNhYKRHYsiwEkVsO4huDa7ABn8rSjU0/0oEYeGU5OZiSkuPX7ryFESE+/rcx2WfX8ajcx/lycFP6mzjpig0DtoMsW5ljIHcnbB3LeSste73rrFqED+/6/FksSbBxaRDdJp9n378XpfPUM2EVxOBiIwAngGcwKvGmMcqHL8ZuBUoAfKA8caYld6KxxUXCyUllBw6hCu6+m/47aLbcWfPO3ly4ZN8sOYDruh4hbdCUw1JBCKSrZtnggDrQjx711mX8dy/ybo/sMlaOuPInhPLBkdbCSGqpTXnIbKlNWciMtV6HBKjTU6qSfBaIhARJ/A8MAzIBuaLyOQKf+jfM8a8ZJcfBTwFjPBWTK44q7OwZO/eGiUCgGs7X8u8XfN4bP5jtIpoRf/k/t4KTzUGQZGQmmXdKirIgwObrcSwf9Px+90rYe00KD52Ynl3iEdisJNDRAqEJ9q3JCuZaLJQPubNGkEfYL0xZiOAiLwPjAbKE4ExxvNq6aGAV6924rQTQfHevQRmZNToOQ5x8PdBf+eGqTfwmxm/4eXhL9M9vrs3w1SNVWCYNQIpsZLlR4yBo/vg0DZrSY2y28Gt1v2u5SfXKMAa4RSW6JEcPJJEWIvj98HR1mQ8pbzAm4kgBdjmsZ0N9K1YSERuBX4LBABnV3YiERkPjAdo1apVnQNyxcUDViKojcjASF469yWu/+p6xk0bxxODn+Cs1LPqHIdqhkSs/ojQOGvCW2WK8iF3B+TutvoocndB3i7rPncn5Kyxlt4oqOQ60eKAkFhrme/QOAiNP34fEmtvxx+PIShKaxqqxnzeWWyMeR54XkSuAh4ArqukzMvAywBZWVl1rjW44stqBPtq/dyEkATeOv8tbvn6Fm7/9nbGdxvPr7r9CpfD5z9C1VS4gyCmjXWrSuHRExNEXg4cyYGje61RUEf2ws4l1nZ+JUkDwOGyEkRwdCW3KOs+KOrkY4ERWvPwQ978K7YdaOmxnWrvO5X3gRe9GA+OsDAkMJDiPZVU0WsgLjiOCSMm8Nef/spLS17ih+0/8Me+f9SVSlX9CgipWcIAa62m8gSRYzVPHcmxto/uta4od+ygNTt751Jru6iKZVbEcTxBBEVay4sHhltDbMsfR1R4HHn8cWA4BIRqbaSJ8WYimA9kiEg6VgIYC1zlWUBEMowx6+zNC4B1eJGIENC2DQVrVtf5HCHuEP468K8MSB7AP+f/kyu/uJLz08/nxi430iGmQz1Gq1QNuAKOj4CqqeICKznkH7QTRWU3+3j+YatmUpBrPS6seokWwLogUcWEERBqdZ4HhFmPT7qFVV3GGaDJxYu8lgiMMcUichswFWv46OvGmBUi8giwwBgzGbhNRM4FioADVNIsVN+Cu3bj8Jdf1mguQVVGthnJWaln8eqyV5m4eiJTNk2hb2JfRrcbzTmtztElrFXj5QqE8BbWrbZKS6Awz0oKBYePJ4gC+1bp/lw4uh+KsqHwiPX8wiPHlwSpCYfreMJwh4A72L4PAlewvR0MrqDjj93B9rEgq6znsVPuD/LLhCPGeHWgTr3LysoyCxYsqPPzD378MTvvf4D0zyYR1KF+vsEfKjjEB2s+4ON1H7M9bzvBrmD6JPZhYMpABiQPIDU8VSejKVVRSZGdGI4cTxBFR09MFoVHPR57ljtmDdctyq/w+CgU2/vqOgjRFWTfAq0Vb10B1rYzwNp30n77sTPQ43iAxzmqeV7ZucvKOd3Ht+txnTMRWWiMqWRctB8mguJ9+1g3ZCgxV11Ji/vuq8fIwBjD4j2L+XLTl8zePpvteVaXSFxwHF3jutI1riudYzvTNqotCSEJmhyU8hZjrCawkxJEhWRRaRI5aj+3wKq1lD/23Jdv9c+UFFj3xfnHy5YU1N/7EIedFOwEMewROOPqup2qikTgd0NeXLGxRJx3Hgf++wHRV19NwGkMR61IROjZoic9W/TEGMOWw1uYs3MOS3OWsmzvMmZsm1FeNswdRpvINrSJakPL8JYkhSaRHJZMSlgK8cHxuuKpUqdDxG76CYLgBn5tYzySQlnSqCyplCWSsqRSdiuy9pcUeZzHfhyT7pWQ/a5GAFC0axcbR43GFRtL67fexBUfX0/RVe1QwSFW71/NxkMb2Xhwo3V/aCN7j504r8ElLlqEtiAhJIG44DhigmKIC44jLjiO2KBY6z44lqjAKIJdwVqzUEpVS5uGKnF0/ny2jhuPMzycpL/9jbBBA+shurrJL85n55Gd7MjbwY4jO9iZt5MdR3aQczSHfcf2sTd/L4cqm2QEuB1uIgMjiQyIJDIwkojACKICo8q3IwMjiQiIINQdesItzB1GqDsUt17ARSm/oIngFPLXrGH7HXdSuGULoQMHEv3LqwkbNAhxNr5mmaKSIvbl77MSw7G97D22l0OFhzhUYN0OFx4uf1y2/1jFtW8qEeAIODlJBIQR6golxB1CoDOQIFcQQa4ggp3B5Y9P2nYGEewKLi8f7ArG7XBrbUWpRkITQRVKCws58PY77Hv9dUr27cMZG0vogDMJ7defoE4dCWjbFkdAQL29XkMqLCksTxB5RXkcKTpy0i2vKI+jRUcrPX6s+Bj5xfnkF+dTWFqLoX42QQhwBhDgCLDu7Zvb4T5hv9vpJsARQKAzsMrjZcecDicuceFyuE7aPuFm73M6nFZZcZc/rljG5XDp9alVs6aJoAZMYSG5M2aSO20aR378kZIDB6wDLhfulGTcCS1wJSbibpGAIzISZ1gYjrBwHOFhOMPDkcAgJMCNIyAAcbuRgIDjN7cbnM4m/e24pLSEgpICKzmU5JcnCM/tyo4VlRZRWFJIYWkhhSWFFJUUlT8uLLW2C0oKyh+fUNZ+bokpaZD36BAHTrEShVOsz8spzvL9J9w7TrHfvq+qTPnxas7pEAeCnHgvUu3jsoTmEAcOqngOgkgdzo8DBBw4TnnOst91KfvP43e/vAwnlgPK91csZ+864Xxl5er6vLLjFeOoSbye+0/5PJGT9nu+z6q2T3js8R48P9/a0kRQS6a0lMLNmylYs4b81Wso2raNot27KbZvpqiobicWAYfDmshm38TeV54oKu53OPD43TjhF6X8nKfarvJYhUPU9HnVvL43GEP5f8bYo8Ote+v3195XXq7scXkpjDnF41M8r/xY2f9NDfdVOEeV++xYqPB6Zec9fuTER+bE/yk/UnrDZZx30yN1eq4OH60lcTgIbNOGwDZtiDj//BOOGWMw+fmU5OZSmpdHaW4uJbl5mMICTGGhdSsqKn9cat9TasCUYkpKobTUelxqoLQUU1piHS8txZjS44/L9h9/cSoEUyFyjz8ZJ5Wt4nmmiteo6TmVD5UlpgpJxtp50rbH1gmJ0HM/drI0lXz+J5Q1nPRcz9+T43mu0nR28j5DhaOneJ6pZN+pzm1OeaSaGE/xPFP5q570yFTzelX8A/L8guApNtk7y9hoIqglEUGCg3EEB0NCgq/DUUqp06a9Y0op5ec0ESillJ/TRKCUUn5OE4FSSvk5TQRKKeXnNBEopZSf00SglFJ+ThOBUkr5uSa3xISI5ABb6vj0OGBvtaWaF33P/kHfs384nffc2hhT6cVXmlwiOB0isuBUa200V/qe/YO+Z//grfesTUNKKeXnNBEopZSf87dE8LKvA/ABfc/+Qd+zf/DKe/arPgKllFIn87cagVJKqQo0ESillJ/zm0QgIiNEZI2IrBeRe30dT30RkZYiMkNEVorIChG5094fIyLTRWSdfR9t7xcRedb+OSwVkZ6+fQd1IyJOEVksIv+zt9NF5Cf7ff1XRALs/YH29nr7eJpPA68jEYkSkY9EZLWIrBKR/n7wGd9l/04vF5GJIhLUHD9nEXldRPaIyHKPfbX+bEXkOrv8OhG5rjYx+EUiEBEn8DxwPpAJXCkimb6Nqt4UA78zxmQC/YBb7fd2L/CNMSYD+MbeButnkGHfxgMvNnzI9eJOYJXH9j+Ap40x7YADwI32/huBA/b+p+1yTdEzwFfGmI5Ad6z33mw/YxFJAe4AsowxXQAnMJbm+TlPAEZU2Ferz1ZEYoA/A32BPsCfy5JHjRhjmv0N6A9M9di+D7jP13F56b1+BgwD1gBJ9r4kYI39+D/AlR7ly8s1lRuQav/jOBv4HyBYsy1dFT9vYCrQ337sssuJr99DLd9vJLCpYtzN/DNOAbYBMfbn9j/gvOb6OQNpwPK6frbAlcB/PPafUK66m1/UCDj+S1Um297XrNjV4TOAn4AWxpid9qFdQAv7cXP4WfwL+D1Qam/HAgeNMcX2tud7Kn+/9vFDdvmmJB3IAd6wm8NeFZFQmvFnbIzZDjwBbAV2Yn1uC2nen7On2n62p/WZ+0siaPZEJAz4GPiNMeaw5zFjfUVoFuOEReRCYI8xZqGvY2lALqAn8KIx5gzgCMebCoDm9RkD2M0ao7GSYDIQysnNJ36hIT5bf0kE24GWHtup9r5mQUTcWEngXWPMJ/bu3SKSZB9PAvbY+5v6z2IAMEpENgPvYzUPPQNEiYjLLuP5nsrfr308EtjXkAHXg2wg2xjzk739EVZiaK6fMcC5wCZjTI4xpgj4BOuzb86fs6fafran9Zn7SyKYD2TYIw4CsDqdJvs4pnohIgK8BqwyxjzlcWgyUDZy4DqsvoOy/dfaow/6AYc8qqCNnjHmPmNMqjEmDetz/NYYczUwA/iFXazi+y37OfzCLt+kvjkbY3YB20Skg73rHGAlzfQztm0F+olIiP07Xvaem+3nXEFtP9upwHARibZrU8PtfTXj606SBuyMGQmsBTYA9/s6nnp8XwOxqo1LgZ/t20is9tFvgHXA10CMXV6wRlBtAJZhjcrw+fuo43sfAvzPftwGmAesBz4EAu39Qfb2evt4G1/HXcf32gNYYH/Ok4Do5v4ZAw8Dq4HlwNtAYHP8nIGJWP0gRVi1vxvr8tkC/2e///XADbWJQZeYUEopP+cvTUNKKaVOQROBUkr5OU0ESinl5zQRKKWUn9NEoJRSfk4TgfI5ETEi8qTH9t0i8lA9nXuCiPyi+pKn/TqX2auCzqiwP61sVUkR6SEiI+vxNaNE5BaP7WQR+ai+zq/8hyYC1RgUAJeKSJyvA/HkMYO1Jm4ExhljhlZRpgfWHI/6iiEKKE8ExpgdxhivJz3V/GgiUI1BMda1WO+qeKDiN3oRybPvh4jIdyLymYhsFJHHRORqEZknIstEpK3Hac4VkQUistZeq6jsegaPi8h8e133X3mcd5aITMaayVoxnivt8y8XkX/Y+x7Emtj3mog8XtkbtGe0PwJcISI/i8gVIhJqr0U/z15MbrRd9noRmSwi3wLfiEiYiHwjIovs1x5tn/YxoK19vscr1D6CROQNu/xiERnqce5PROQrsdat/6fHz2OC/b6WichJn4VqvmrzjUcpb3oeWFr2h6mGugOdgP3ARuBVY0wfsS7OczvwG7tcGtYa7W2BGSLSDrgWa3p+bxEJBH4QkWl2+Z5AF2PMJs8XE5FkrHXue2GthT9NRC42xjwiImcDdxtjFlQWqDGm0E4YWcaY2+zz/Q1rKYT/E5EoYJ6IfO0RQzdjzH67VnCJMeawXWuaayeqe+04e9jnS/N4yVutlzVdRaSjHWt7+1gPrFVqC4A1IvJvIAFIMdba/9jxKD+hNQLVKBhrxdS3sC5GUlPzjTE7jTEFWFPuy/6QL8P641/mA2NMqTFmHVbC6Ii1Fsu1IvIz1rLdsVgX+wCYVzEJ2HoDM421EFox8C5wVi3irWg4cK8dw0ysZRJa2cemG2P2248F+JuILMVabiCF48sSn8pA4B0AY8xqYAtQlgi+McYcMsbkY9V6WmP9XNqIyL9FZARwuJJzqmZKawSqMfkXsAh4w2NfMfYXFhFxAAEexwo8Hpd6bJdy4u92xXVUDNYf19uNMScszCUiQ7CWeW4IAowxxqypEEPfCjFcDcQDvYwxRWKtvBp0Gq/r+XMrwbrQywER6Y518Zebgcux1q5RfkBrBKrRsL8Bf8Dxyw8CbMZqigEYBbjrcOrLRMRh9xu0wbqq01Tg12It4Y2ItBfrYi9VmQcMFpE4sS5/eiXwXS3iyAXCPbanAreLiNgxnHGK50ViXYOhyG7rb32K83mahZVAsJuEWmG970rZTU4OY8zHwANYTVPKT2giUI3Nk4Dn6KFXsP74LsG6NGFdvq1vxfojPgW42W4SeRWrWWSR3cH6H6qpIRtrud97sZZCXgIsNMZ8VtVzKpgBZJZ1FgOPYiW2pSKywt6uzLtAlogsw+rbWG3Hsw+rb2N5JZ3ULwAO+zn/Ba63m9BOJQWYaTdTvYN1OVflJ3T1UaWU8nNaI1BKKT+niUAppfycJgKllPJzmgiUUsrPaSJQSik/p4lAKaX8nCYCpZTyc/8PhjNGDz1U0EkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = i)\n",
    "\n",
    "print(\"For the train set:\")\n",
    "for learning_rate in (0.0001, 0.001, 0.01, 0.1):\n",
    "    OlsGd_object = OlsGd(learning_rate, verbose = False, early_stop = False)\n",
    "    OlsGd_object._fit(X_train, y_train)\n",
    "    predicted_y = OlsGd_object._predict(X_train)\n",
    "    print(\"MSE: \", OlsGd_object.score(X_train, y_train))\n",
    "    OlsGd_object.plot()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "For the test set:\n",
      "MSE:  73.8356969071403\n",
      "MSE:  39.69581485473349\n",
      "MSE:  34.804438566952406\n",
      "MSE:  34.674681355514444\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"For the test set:\")\n",
    "for learning_rate in (0.0001, 0.001, 0.01, 0.1):\n",
    "    OlsGd_object = OlsGd(learning_rate, verbose = False, early_stop = False)\n",
    "    OlsGd_object._fit(X_train, y_train)\n",
    "    predicted_y = OlsGd_object._predict(X_test)\n",
    "    print(\"MSE: \", OlsGd_object.score(X_test, y_test))\n",
    "    OlsGd_object.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. What is the effect of learning rate? \n",
    "* If the learning rate is too low, we might not achive the minimum because we are approaching it very slowly. If it's too high, we might skip it and not find the minimum. \n",
    "2. How would you find number of iteration automatically?\n",
    "* We can set up a delta to calculate the difference in the loss between two iterations and if this delta is low (the loss converges), we will stop."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7HVfnXvZFi98"
   },
   "source": [
    "## Exercise 2 - Ridge Linear Regression\n",
    "\n",
    "Recall that ridge regression is identical to OLS but with a L2 penalty over the weights:\n",
    "\n",
    "$L(y,\\hat{y})=\\sum_{i=1}^{i=N}{(y^{(i)}-\\hat{y}^{(i)})^2} + \\lambda \\left\\Vert w \\right\\Vert_2^2$\n",
    "\n",
    "where $y^{(i)}$ is the **true** value and $\\hat{y}^{(i)}$ is the **predicted** value of the $i_{th}$ example, and $N$ is the number of examples\n",
    "\n",
    "* Show, by differentiating the above loss, that the analytical solution is $w_{Ridge}=(X^TX+\\lambda I)^{-1}X^Ty$\n",
    "* Change `OrdinaryLinearRegression` and `OrdinaryLinearRegressionGradientDescent` classes to work also for ridge regression (do not use the random noise analogy but use the analytical derivation). Either add a parameter, or use inheritance.\n",
    "* **Bonus: Noise as a regularizer**: Show that OLS (ordinary least square), if one adds multiplicative noise to the features the **average** solution for $W$ is equivalent to Ridge regression. In other words, if $X'= X*G$ where $G$ is an uncorrelated noise with variance $\\sigma$ and mean 1, then solving for $X'$ with OLS is like solving Ridge for $X$. What is the interpretation? \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "class RidgeLs(Ols):\n",
    "    def __init__(self, ridge_lambda, *wargs, **kwargs):\n",
    "        super(RidgeLs,self).__init__(*wargs, **kwargs)\n",
    "        self.ridge_lambda = ridge_lambda\n",
    "        self.ridge = True\n",
    "\n",
    "    def _fit(self, X, Y):\n",
    "    #Closed form of ridge regression\n",
    "        super()._fit(X, Y, ridge = self.ridge)\n",
    "        \n",
    "    def _predict(self, X):\n",
    "    #Closed form of ridge regression\n",
    "        return super()._predict(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training MSE: 22.28556658395527\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Predicted Y')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "ridge_lambda = 0.5\n",
    "RidgeLs_object = RidgeLs(ridge_lambda)\n",
    "RidgeLs_object._fit(X, y)\n",
    "predicted_y = RidgeLs_object._predict(X)\n",
    "print(\"training MSE:\", RidgeLs_object.score(X, y))\n",
    "plt.scatter(y,predicted_y)\n",
    "plt.xlabel('Y Train')\n",
    "plt.ylabel('Predicted Y')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average MSE for train is: 21.742592586769142\n",
      "Average MSE for test is: 25.81376687764531\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Ttest_relResult(statistic=-2.989547270265569, pvalue=0.007534023902857832)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mse_train = []\n",
    "mse_test = []\n",
    "\n",
    "for i in range(20):\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = i)\n",
    "    RidgeLs_object = RidgeLs(ridge_lambda)\n",
    "    RidgeLs_object._fit(X_train,y_train)\n",
    "    mse_train.append(RidgeLs_object.score(X_train, y_train))\n",
    "    mse_test.append(RidgeLs_object.score(X_test, y_test))\n",
    "\n",
    "print(\"Average MSE for train is:\",np.mean(mse_train))\n",
    "print(\"Average MSE for test is:\", np.mean(mse_test))\n",
    "stats.ttest_rel(np.array(mse_train),np.array(mse_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "class RidgeLsGd(OlsGd):\n",
    "    def __init__(self, ridge_lambda, *wargs, **kwargs):\n",
    "        super(RidgeLsGd,self).__init__(*wargs, **kwargs)\n",
    "        self.ridge_lambda = ridge_lambda\n",
    "        self.ridge = True\n",
    "\n",
    "    def _fit(self, X, Y):\n",
    "    #Closed form of ridge regression\n",
    "        super()._fit(X, Y, ridge = self.ridge, ridge_lambda = self.ridge_lambda)\n",
    "        \n",
    "    def _predict(self, X):\n",
    "    #Closed form of ridge regression\n",
    "        return super()._predict(X)\n",
    "    \n",
    "    def plot(self):\n",
    "        plt.plot(self.iterations, self.loss_history, label=f'Alpha: {self.learning_rate}')\n",
    "        plt.xlabel(\"Number of Iterations\")\n",
    "        plt.ylabel(\"Loss\") \n",
    "        plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MSE: 44.81389962644876\n",
      "MSE: 23.973690030862574\n",
      "MSE: 21.95697759488851\n",
      "MSE: 21.895094900243485\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "ridge_lambda = 0.5\n",
    "#Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.\n",
    "for learning_rate in (0.0001, 0.001, 0.01, 0.1):\n",
    "    RidgeLsGd_object = RidgeLsGd(ridge_lambda, learning_rate)\n",
    "    RidgeLsGd_object._fit(X, y)\n",
    "    predicted_y = RidgeLsGd_object._predict(X)\n",
    "    print(\"MSE:\", RidgeLsGd_object.score(X, y))\n",
    "    RidgeLsGd_object.plot()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Use scikitlearn implementation for OLS, Ridge and Lasso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LinearRegression, Lasso, Ridge\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "21.21445744330387"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "metrics.mean_squared_error(lr.predict(X_test), y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "25.895850181992028"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lasso = Lasso()\n",
    "lasso.fit(X_train, y_train)\n",
    "metrics.mean_squared_error(lasso.predict(X_test), y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "21.19185044048607"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ridge = Ridge()\n",
    "ridge.fit(X_train, y_train)\n",
    "metrics.mean_squared_error(ridge.predict(X_test), y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Regression & Regularization - Exercise.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
