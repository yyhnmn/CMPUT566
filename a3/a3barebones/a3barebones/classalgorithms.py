import numpy as np

import MLCourse.utilities as utils
from sklearn.model_selection import KFold

# Susy: ~50 error


class Classifier:
    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

# Susy: ~27 error


class LinearRegressionClass(Classifier):
    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + \
            self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

# Susy: ~25 error


class NaiveBayes(Classifier):
    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items(
            {'usecolumnones': False}, parameters)

    def learn(self, Xtrain, ytrain):
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
        else:
            raise Exception('Can only handle binary classification')
        numsamples = Xtrain.shape[0]
        numfeatures = Xtrain.shape[1]
        self.class_mean = np.zeros((self.numclasses, numfeatures))
        self.class_std = np.zeros((self.numclasses, numfeatures))
        self.class_0 = []
        self.class_1 = []
        for i in range(numsamples):
            if ytrain[i] == 0:
                self.class_0.append(Xtrain[i])
            else:
                self.class_1.append(Xtrain[i])
        self.class_0 = np.asarray(self.class_0)
        self.class_1 = np.asarray(self.class_1)

        self.class_mean[0] = np.mean(
            self.class_0, axis=0).reshape(1, numfeatures)
        self.class_mean[1] = np.mean(
            self.class_1, axis=0).reshape(1, numfeatures)

        self.class_std[0] = np.std(
            self.class_0, axis=0).reshape(1, numfeatures)
        self.class_std[1] = np.std(
            self.class_1, axis=0).reshape(1, numfeatures)

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]

        predictions = []
        if not self.getparams()['usecolumnones']:
            Xtest = Xtest[:, :-1]
        numfeatures = Xtest.shape[1]
        for i in range(numsamples):
            prob_0 = 1
            prob_1 = 1
            for j in range(numfeatures):
                prob_0 *= utils.gaussian_pdf(Xtest[i][j],
                                             self.class_mean[0][j], self.class_std[0][j])
                prob_1 *= utils.gaussian_pdf(Xtest[i][j],
                                             self.class_mean[1][j], self.class_std[1][j])
            if prob_0 < prob_1:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.reshape(predictions, [numsamples, 1])

# Susy: ~23 error


class LogisticReg(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items(
            {'stepsize': 0.01, 'epochs': 100}, parameters)
        self.weights = None

    def learn(self, X, y):
        # SGD implementation of LogisticReg
        numsamples = X.shape[0]
        numfeatures = X.shape[1]
        stepsize_init = self.getparams()['stepsize']
        self.weights = np.random.rand(numfeatures)
        datapts = np.arange(numsamples)
        for i in range(self.getparams()['epochs']):
            np.random.shuffle(datapts)
            for j in datapts:
                gradient = (utils.sigmoid(X[datapts[j]].T.dot(
                    self.weights))-y[datapts[j]])*(X[datapts[j]])
                stepsize = stepsize_init / (i+1)
                self.weights = np.subtract(self.weights, stepsize*gradient)

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return np.reshape(ytest, [numsamples, 1])


# BGD training
# Susy: ~23 error (4 hidden units)
class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.001,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception(
                'NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None

    def learn(self, Xtrain, ytrain):
        
        numfeatures = Xtrain.shape[1]
        self.nh = self.getparams()['nh']
        self.stepsize_init = self.getparams()['stepsize']

        self.wi = np.random.rand(numfeatures,self.nh)
        self.wo = np.random.rand(self.nh,1)

        for i in range(self.getparams()['epochs']):
            ah,ao = self.evaluate(Xtrain)
            stepsize = self.stepsize_init
            delta1 = ao - ytrain
            g_wo = np.dot(ah.T,delta1)
            delta2 = np.dot(delta1,self.wo.T)*ah*(1-ah)
            g_wi = np.dot(Xtrain.T,delta2)
            self.wi = np.subtract(self.wi, stepsize*g_wi)
            self.wo = np.subtract(self.wo, stepsize*g_wo)

    def predict(self, Xtest):
        ytest = self.evaluate(Xtest)[1]
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest

    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(inputs, self.wi))

        # output activations
        ao = self.transfer(np.dot(ah,self.wo))

        return (
            ah,  
            ao,  
        )

# # SGD training, takes long time to train
# # Susy: ~23 error (4 hidden units)
# class NeuralNet(Classifier):
#     def __init__(self, parameters={}):
#         self.params = utils.update_dictionary_items({
#             'nh': 4,
#             'transfer': 'sigmoid',
#             'stepsize': 0.001,
#             'epochs': 10,
#         }, parameters)

#         if self.params['transfer'] is 'sigmoid':
#             self.transfer = utils.sigmoid
#             self.dtransfer = utils.dsigmoid
#         else:
#             # For now, only allowing sigmoid transfer
#             raise Exception(
#                 'NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

#         self.wi = None
#         self.wo = None

#     def learn(self, Xtrain, ytrain):
#         numsamples = Xtrain.shape[0]
#         numfeatures = Xtrain.shape[1]
#         self.nh = self.getparams()['nh']
#         self.stepsize_init = self.getparams()['stepsize']

#         self.wi = np.random.rand(self.nh, numfeatures)/np.sqrt(numfeatures)
#         self.wo = np.random.rand(1, self.nh)/np.sqrt(self.nh)
#         datapts = np.arange(numsamples)
#         for i in range(self.getparams()['epochs']):
#             np.random.shuffle(datapts)
#             for j in datapts:
#                 g_wi, g_wo = self.update(Xtrain[j], ytrain[j])
#                 stepsize = self.stepsize_init/(i+1)
#                 self.wi = np.subtract(self.wi, stepsize*g_wi)
#                 self.wo = np.subtract(self.wo, stepsize*g_wo)

#     def predict(self, Xtest):
#         ytest = self.evaluate(Xtest)[1]
#         ytest[ytest >= 0.5] = 1
#         ytest[ytest < 0.5] = 0
#         return ytest

#     def evaluate(self, inputs):
#         # hidden activations
#         ah = self.transfer(np.dot(self.wi, inputs.T))

#         # output activations
#         ao = self.transfer(np.dot(self.wo, ah)).T

#         return (
#             ah,  # shape: [nh, samples]
#             ao,  # shape: [samples, 1]
#         )

#     def update(self, inputs, outputs):
#         inputs = np.reshape(inputs, [1, len(inputs)])     # [1,features]
#         ah, ao = self.evaluate(inputs)                  # ah[nh,1] ao[1,1]
#         delta1 = ao-outputs                             # [1,1]
#         g_wo = ah.dot(delta1).T                         # [1,nh]

#         delta2 = delta1.dot(self.wo)                    # [1,nh]
#         delta2 = np.multiply(delta2, ah.T)               # [1,nh]
#         delta2 = np.multiply(delta2, (1-ah.T))           # [1,nh]
#         g_wi = inputs.T.dot(delta2).T                   # [nh,features]
#         return(
#             g_wi,
#             g_wo,
#         )


# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)

class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
            'kernal': 'linear'
        }, parameters)
        self.weights = None

    def linearKernal(self, X):
        K = np.zeros((X.shape[0], self.getparams()['centers']))
        for i in range(X.shape[0]):
            for j in range(self.getparams()['centers']):
                K[i][j] = self.linear(X[i], self.centers[j])
        return K

    def linear(self, x, c):
        k = 0
        for i in range(x.shape[0]):
            k = k+x[i]*c[i]
        return k

    def hammingKernal(self, X):
        K = np.zeros((X.shape[0], self.getparams()['centers']))
        for i in range(X.shape[0]):
            for j in range(self.getparams()['centers']):
                K[i][j] = self.hammingDistance(X[i], self.centers[j])
        return K

    def hammingDistance(self, a, b):
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    def learn(self, X, y):
        K = None
        row_rand_array = np.arange(X.shape[0])
        np.random.shuffle(row_rand_array)
        self.centers = X[row_rand_array[0:self.getparams()['centers']]]
        if self.getparams()['kernal'] == 'linear':
            K = self.linearKernal(X)
        elif self.getparams()['kernal'] == 'hamming':
            K = self.hammingKernal(X)
        self.weights = np.zeros(K.shape[1])
        for i in range(self.getparams()['epochs']):
            datapts = np.arange(X.shape[0])
            np.random.shuffle(datapts)
            for j in datapts:
                gradient = (utils.sigmoid(K[j].dot(
                    self.weights))-y[j])*(K[j])
                stepsize = self.getparams()['stepsize'] / (i+1)
                self.weights = np.subtract(self.weights, stepsize*gradient)

    def predict(self, Xtest):
        if self.getparams()['kernal'] == 'linear':
            K = self.linearKernal(Xtest)
        elif self.getparams()['kernal'] == 'hamming':
            K = self.hammingKernal(Xtest)
        ytest = utils.sigmoid(np.dot(K, self.weights))
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return np.reshape(ytest, [Xtest.shape[0], 1])

# hidden layer has same number of nodes


class TwoHiddenLayerNeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.001,
            'epochs': 100,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception(
                'NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi1 = None
        self.wi2 = None
        self.wo = None
    # SGD implementation
    # def learn(self, Xtrain, ytrain):
    #     numsamples = Xtrain.shape[0]
    #     numfeatures = Xtrain.shape[1]
    #     self.nh = self.getparams()['nh']
    #     self.stepsize_init = self.getparams()['stepsize']

    #     self.wi1 = np.random.rand(self.nh, numfeatures)/np.sqrt(numfeatures)
    #     self.wi2 = np.random.rand(self.nh,self.nh)/np.sqrt(self.nh)
    #     self.wo = np.random.rand(1, self.nh)/np.sqrt(self.nh)
    #     datapts = np.arange(numsamples)
    #     for i in range(self.getparams()['epochs']):
    #         np.random.shuffle(datapts)
    #         for j in datapts:
    #             g_wi1,g_wi2,g_wo = self.update(Xtrain[j], ytrain[j])
    #             stepsize = self.stepsize_init/(i+1)
    #             self.wi1 = np.subtract(self.wi1, stepsize*g_wi1)
    #             self.wi2 = np.subtract(self.wi2, stepsize*g_wi2)
    #             self.wo = np.subtract(self.wo, stepsize*g_wo)

    # Adam implementation
    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        numfeatures = Xtrain.shape[1]
        self.nh = self.getparams()['nh']
        self.wi1 = np.random.rand(self.nh, numfeatures)/np.sqrt(numfeatures)
        self.wi2 = np.random.rand(self.nh, self.nh)/np.sqrt(self.nh)
        self.wo = np.random.rand(1, self.nh)/np.sqrt(self.nh)

        iterations = self.getparams()['epochs']
        alpha = self.getparams()['stepsize']
        beta1 = 0.9
        beta2 = 0.999
        err = 10e-8
        mt1 = 0
        vt1 = 0

        mt2 = 0
        vt2 = 0

        mto = 0
        vto = 0
        t = 0

        datapts = np.arange(numsamples)

        while t < iterations:
            t += 1
            np.random.shuffle(datapts)
            for j in datapts:
                gt1, gt2, gto = self.update(Xtrain[j], ytrain[j])
                mt1 = beta1 * mt1 + (1-beta1) * gt1
                vt1 = beta2 * vt1 + (1-beta2) * pow(gt1, 2)
                mt2 = beta1 * mt2 + (1-beta1) * gt2
                vt2 = beta2 * vt2 + (1-beta2) * pow(gt2, 2)
                mto = beta1 * mto + (1-beta1) * gto
                vto = beta2 * vto + (1-beta2) * pow(gto, 2)
                mrt1 = mt1/(1-beta1**t)
                vrt1 = vt1/(1-beta2**t)
                mrt2 = mt2/(1-beta1**t)
                vrt2 = vt2/(1-beta2**t)
                mrto = mto/(1-beta1**t)
                vrto = vto/(1-beta2**t)
                self.wi1 = np.subtract(self.wi1, np.divide(
                    alpha*mrt1, np.add(np.sqrt(vrt1), err)))
                self.wi2 = np.subtract(self.wi2, np.divide(
                    alpha*mrt2, np.add(np.sqrt(vrt2), err)))
                self.wo = np.subtract(self.wo, np.divide(
                    alpha*mrto, np.add(np.sqrt(vrto), err)))

    def predict(self, Xtest):
        ytest = self.evaluate(Xtest)[2]
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest

    def evaluate(self, inputs):
        # hidden activations
        ah1 = self.transfer(np.dot(self.wi1, inputs.T))
        ah2 = self.transfer(np.dot(self.wi2, ah1))
        # output activations
        ao = self.transfer(np.dot(self.wo, ah2)).T

        return (
            ah1,
            ah2,
            ao,
        )

    def update(self, inputs, outputs):
        inputs = np.reshape(inputs, [1, len(inputs)])

        ah1, ah2, ao = self.evaluate(inputs)
        delta1 = ao-outputs
        g_wo = ah2.dot(delta1).T

        delta2 = delta1.dot(self.wo)
        delta2 = np.multiply(delta2, ah2.T)
        delta2 = np.multiply(delta2, (1-ah2.T))

        g_wi2 = ah1.dot(delta2).T

        delta3 = delta2.dot(self.wi2)
        delta3 = np.multiply(delta3, ah1.T)
        delta3 = np.multiply(delta3, (1-ah1.T))

        g_wi1 = inputs.T.dot(delta3).T

        return(
            g_wi1,
            g_wi2,
            g_wo,
        )
