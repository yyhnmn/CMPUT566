import numpy as np
import math

import MLCourse.utilities as utils
import script_regression as script
import matplotlib.pyplot as plt
import time

# -------------
# - Baselines -
# -------------


class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__(self, parameters={}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest


class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__(self, parameters={}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest


class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """

    def __init__(self, parameters={}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """

    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.0,
            'features': [1, 2, 3, 4, 5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        #numfeatures = Xless.shape[1]
        #inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        inner = (Xless.T.dot(Xless) / numsamples)
        self.weights = np.linalg.pinv(inner).dot(
            Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

# ---------
# - TODO: -
# ---------


class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """

    def __init__(self, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items(
            {'regwgt': 0.01}, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        inner = (Xtrain.T.dot(Xtrain) / numsamples) + \
            self.params['regwgt'] * np.identity(Xtrain.shape[1])
        self.weights = np.linalg.pinv(inner).dot(
            Xtrain.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest


class LassoRegression(Regressor):
    '''
    An iterative solution approach that uses the soft thresholding operator
    '''

    def __init__(self, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items(
            {'regwgt': 0.01}, parameters)

    # page 76 in notes
    def prox(self, stepsize, w, regwgt):
        for i in range(w.shape[0]):
            if w[i] > stepsize*regwgt:
                self.weights[i] = w[i] - stepsize*regwgt
            elif np.abs(w[i]) <= stepsize*regwgt:
                self.weights[i] = 0
            elif w[i] < -stepsize*regwgt:
                self.weights[i] = w[i] + stepsize*regwgt

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros(Xtrain.shape[1])
        err = math.inf
        tolerance = 10e-4
        max_iterations = 10e5
        XX = (Xtrain.T.dot(Xtrain)) / numsamples
        Xy = (Xtrain.T.dot(ytrain)) / numsamples
        norm = np.linalg.norm(XX)
        stepsize = 1/(2*norm)
        c_w = script.geterror_squared(Xtrain.dot(self.weights), ytrain)
        iteration = 0
        while np.abs(c_w - err) > tolerance and iteration < max_iterations:
            iteration += 1
            err = c_w
            prox_operator = np.add(np.subtract(
                self.weights, stepsize*(XX.dot(self.weights))), stepsize * Xy)
            self.prox(stepsize, prox_operator, self.params['regwgt'])
            c_w = script.geterror_squared(Xtrain.dot(self.weights), ytrain)
       
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest


class SGD(Regressor):
    '''
    stochastic gradient descent approach to obtaining the linear regression solution
    '''

    def __init__(self, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = parameters

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        epoches = 1000
        stepsize_init = 0.01
        numfeatures = Xtrain.shape[1]
        self.weights = np.random.rand(numfeatures)
        datapts = np.arange(numsamples)
        y = np.zeros(epoches)
        times = []
        start = time.time()
        for i in range(epoches):
            np.random.shuffle(datapts)
            for j in datapts:
                gradient = (Xtrain[datapts[j]].T.dot(
                    self.weights) - ytrain[datapts[j]]) * Xtrain[datapts[j]]
                stepsize = stepsize_init/(i+1)
                self.weights = np.subtract(self.weights, stepsize*gradient)
            y[i] = script.geterror(
                np.dot(Xtrain, self.weights), ytrain)
            times.append(time.time()-start)

        '''
        code below is to plot error vs number of epoches for SGD
        '''
        # x = np.arange(epoches)
        # plt.plot(x, y)
        # plt.xlabel('Number of epoches')
        # plt.ylabel('Error')
        # plt.title('Error vs Number of epoches for SGD')
        # plt.show()
        '''
        plot error vs runtime
        '''
        # plt.plot(times, y)
        # plt.xlabel('runtime(s)')
        # plt.ylabel('Error')
        # plt.title('Error vs runtime for SGD')
        # plt.show()

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest


class BGD(Regressor):
    '''
    batch gradient descent with line search
    '''

    def __init__(self, parameters={}):
         # Default parameters, any of which can be overwritten by values passed to params
        self.params = parameters

    def lineSearch(self, Xtrain, ytrain, gradient, cost):
        stepsize_max = 1.0
        t = 0.7
        tolerance = 10e-4
        stepsize = stepsize_max
        weight = self.weights
        obj = cost
        max_interations = 100
        iteration = 0

        while iteration < max_interations:
            weight = self.weights - stepsize * gradient
            if cost < obj-tolerance:
                break
            else:
                stepsize = t * stepsize
            cost = script.geterror_squared(np.dot(Xtrain, weight), ytrain)
            iteration += 1
        if iteration == max_interations:
            stepsize = 0
            return stepsize
        return stepsize

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        numfeatures = Xtrain.shape[1]
        self.weights = np.random.rand(numfeatures)
        iteration = 0
        err = math.inf
        tolerance = 10e-4
        c_w = script.geterror_squared(Xtrain.dot(self.weights), ytrain)
        max_iterations = 100000
        y = []
        times = []
        start = time.time()
        while np.abs(c_w-err) > tolerance and iteration < max_iterations:
            err = c_w
            gradient = Xtrain.T.dot(np.subtract(
                np.dot(Xtrain, self.weights), ytrain)) / numsamples
            stepsize = self.lineSearch(Xtrain, ytrain, gradient, c_w)
            self.weights = self.weights - stepsize*gradient
            c_w = script.geterror_squared(Xtrain.dot(self.weights), ytrain)
            y.append(script.geterror(Xtrain.dot(self.weights), ytrain))
            times.append(time.time()-start)
            iteration += 1

        # '''
        # plot error vs epoches for BGD
        # '''
        # x = np.arange(iteration)
        # plt.plot(x, y)
        # plt.xlabel('Number of epoches')
        # plt.ylabel('Error')
        # plt.title('Error vs Number of epoches for BGD')
        # plt.show()

        '''
        plot error vs runtime
        '''
        # plt.plot(times, y)
        # plt.xlabel('runtime(s)')
        # plt.ylabel('Error')
        # plt.title('Error vs runtime for BGD')
        # plt.show()

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest


class Momentum(Regressor):
    def __init__(self, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = parameters

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        iterations = 1000
        alpha = 0.001
        mt = 0.1
        gt = 0
        t = 0
        threshold = 10e-5
        datapts = np.arange(numsamples)
        y = []
        numfeatures = Xtrain.shape[1]
        self.weights = np.random.rand(numfeatures)

        while t < iterations:
            t += 1
            for j in datapts:
                prev_weights = self.weights
                gt = mt*gt + alpha * (Xtrain[j].T.dot(self.weights) - ytrain[j]) * Xtrain[j]
                self.weights = np.subtract(self.weights, gt)
            if np.linalg.norm(np.subtract(self.weights, prev_weights)) <= threshold:
                break
            prev_weights = self.weights

            y.append(script.geterror(
                np.dot(Xtrain, self.weights), ytrain))

        # '''
        # code below is to plot error vs number of iterations for Momentum
        # '''
        # x = np.arange(t)
        # plt.plot(x, y)
        # plt.xlabel('Number of iterations')
        # plt.ylabel('Error')
        # plt.title('Error vs Number of iterations for Momentum')
        # plt.show()

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest


class Adam(Regressor):
    def __init__(self, parameters={}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = parameters

    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        # parameters
        iterations = 1000
        alpha = 0.001
        beta1 = 0.9
        beta2 = 0.999
        err = 10e-8
        mt = 0
        vt = 0
        t = 0
        numfeatures = Xtrain.shape[1]
        self.weights = np.random.rand(numfeatures)
        # change threshold to a larger value will decrease runtime at the cost of accuracy
        threshold = 10e-5
        datapts = np.arange(numsamples)
        y = []

        while t < iterations:
            t += 1
            for j in datapts:
                prev_weights = self.weights
                gt = (Xtrain[j].T.dot(self.weights) -
                      ytrain[j]) * Xtrain[j]
                # Momentum
                mt = beta1 * mt + (1-beta1) * gt
                vt = beta2 * vt + (1-beta2) * pow(gt, 2)
                # Momentum with bias correction
                mrt = mt/(1-beta1**t)
                vrt = vt/(1-beta2**t)

                self.weights = np.subtract(self.weights, np.divide(
                    alpha*mrt, np.add(np.sqrt(vrt), err)))

            if np.linalg.norm(np.subtract(self.weights, prev_weights)) <= threshold:
                break
            prev_weights = self.weights
            y.append(script.geterror(
                np.dot(Xtrain, self.weights), ytrain))

        '''
        code below is to plot error vs number of iterations for Adam
        '''
        # x = np.arange(t)
        # plt.plot(x, y)
        # plt.xlabel('Number of iterations')
        # plt.ylabel('Error')
        # plt.title('Error vs Number of iterations for Adam')
        # plt.show()

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
