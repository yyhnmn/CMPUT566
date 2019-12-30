import numpy as np
import math
import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import classalgorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100


def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))


""" k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test

NOTE: utils.leaveOneOut will likely be useful for this problem.
Check utilities.py for example usage.
"""


def cross_validate(K, X, Y, Algorithm, parameters):
    numsamples = X.shape[0]
    all_errors = np.zeros((len(parameters), K))
    for k in range(K):
        xtrainset = np.delete(
            X, [int(numsamples/K)*k, int(numsamples/K)*(k+1)-1], axis=0)
        ytrainset = np.delete(
            Y, [int(numsamples/K)*k, int(numsamples/K)*(k+1)-1], axis=0)
        xtestset = X[[int(numsamples/K)*k, int(numsamples/K)*(k+1)-1]]
        ytestset = Y[[int(numsamples/K)*k, int(numsamples/K)*(k+1)-1]]
        

        for i, params in enumerate(parameters):
            learner = Algorithm(params)
            learner.learn(xtrainset, ytrainset)
            predictions = learner.predict(xtestset)
            error = geterror(ytestset, predictions)
            all_errors[i][k] = error

    avg_errors = np.mean(all_errors, axis=1)
    std_errors = np.std(all_errors,axis=1)/math.sqrt(K)
    for i, params in enumerate(parameters):
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        print('standard error:', std_errors[i])

    best_parameters = parameters[0]
    min_error = avg_errors[0]
    for i,params in enumerate(parameters):
        if avg_errors[i] < min_error:
            min_error = avg_errors[i]
            best_parameters = params
    return best_parameters

""" stratified k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test
"""
def stratifiedKfold(K,X,Y,Algorithm,parameters):

    all_errors = np.zeros((len(parameters), K))
    sample0 = []
    sample1 = []
    output0 = []
    output1 = []

    # shuffle data
    arrays = np.c_[X,Y]
    np.random.shuffle(arrays)
    X = arrays[:,:-1]
    Y = arrays[:,-1].reshape(len(Y),1)
    
    for i in range(X.shape[0]):
        if Y[i] == 0:
            sample0.append(X[i])
            output0.append(Y[i])
        elif Y[i] == 1:
            sample1.append(X[i])
            output1.append(Y[i])
    sample0 = np.asarray(sample0)
    sample1 = np.asarray(sample1)
    output0 = np.asarray(output0)
    output1 = np.asarray(output1)

    for k in range(K):
        xtrainset0 = np.delete(
            sample0, [int((sample0.shape[0]/K))*k, int((sample0.shape[0]/K))*(k+1)-1], axis=0)
        ytrainset0 = np.delete(
            output0, [int((sample0.shape[0]/K))*k, int((sample0.shape[0]/K))*(k+1)-1], axis=0)
        
        xtrainset1 = np.delete(
            sample1, [int((sample1.shape[0]/K))*k, int((sample1.shape[0]/K))*(k+1)-1], axis=0)
        ytrainset1 = np.delete(
            output1, [int((sample1.shape[0]/K))*k, int((sample1.shape[0]/K))*(k+1)-1], axis=0)
        
        xtestset0 = sample0[[int(sample0.shape[0]/K)*k, int(sample0.shape[0]/K)*(k+1)-1]]
        ytestset0 = output0[[int(sample0.shape[0]/K)*k, int(sample0.shape[0]/K)*(k+1)-1]]

        xtestset1 = sample1[[int(sample1.shape[0]/K)*k, int(sample1.shape[0]/K)*(k+1)-1]]
        ytestset1 = output1[[int(sample1.shape[0]/K)*k, int(sample1.shape[0]/K)*(k+1)-1]]
        

        xtrainset = np.vstack((xtrainset0,xtrainset1))
        ytrainset = np.vstack((ytrainset0,ytrainset1))

        xtestset = np.vstack((xtestset0,xtestset1))
        ytestset = np.vstack((ytestset0,ytestset1))

        for i, params in enumerate(parameters):
            learner = Algorithm(params)
            learner.learn(xtrainset, ytrainset)
            predictions = learner.predict(xtestset)
            error = geterror(ytestset, predictions)
            all_errors[i][k] = error

    avg_errors = np.mean(all_errors, axis=1)
    std_errors = np.std(all_errors,axis=1)/math.sqrt(K)
    for i, params in enumerate(parameters):
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        print('standard error:', std_errors[i])

    best_parameters = parameters[0]
    min_error = avg_errors[0]
    for i,params in enumerate(parameters):
        if avg_errors[i] < min_error:
            min_error = avg_errors[i]
            best_parameters = params
    return best_parameters

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=10,
                        help='Specify the number of runs')

    # can change dataset to census
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset

    classalgs = {
        # 'Random': algs.Classifier,
        'Naive Bayes': algs.NaiveBayes,
        # 'Linear Regression': algs.LinearRegressionClass,
        # 'Logistic Regression': algs.LogisticReg,
        # 'Neural Network': algs.NeuralNet,
        # 'Kernel Logistic Regression': algs.KernelLogisticRegression,
        # 'TwoHiddenLayerNeuralNet': algs.TwoHiddenLayerNeuralNet,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Naive Bayes': [
            # first set of parameters to try
            {'usecolumnones': True},
            # second set of parameters to try
            {'usecolumnones': False},
        ],
        'Logistic Regression': [
            {'stepsize': 0.001},
            {'stepsize': 0.01},
            {'stepsize': 0.05}
        ],
        'Neural Network': [
            {'epochs': 1000, 'nh': 4},
            {'epochs': 1000, 'nh': 8},
            {'epochs': 1000, 'nh': 16},
            {'epochs': 1000, 'nh': 32},
        ],
        'Kernel Logistic Regression': [
            # {'centers': 10, 'stepsize': 0.01,'kernal': 'hamming'},
            # {'centers': 20, 'stepsize': 0.01,'kernal': 'hamming'},
            # {'centers': 40, 'stepsize': 0.01,'kernal': 'hamming'},
            # {'centers': 80, 'stepsize': 0.01,'kernal': 'hamming'},
            {'centers': 10, 'stepsize': 0.01,'kernal': 'linear'},
            {'centers': 20, 'stepsize': 0.01,'kernal': 'linear'},
            {'centers': 40, 'stepsize': 0.01,'kernal': 'linear'},
            {'centers': 80, 'stepsize': 0.01,'kernal': 'linear'},
        ],
        'TwoHiddenLayerNeuralNet': [
            {'epochs': 1000, 'nh': 4},
            {'epochs': 1000, 'nh': 8},
            {'epochs': 1000, 'nh': 16},
            {'epochs': 1000, 'nh': 32},
        ],
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)

    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize, testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        # print(trainset[0])
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        # cast the Y vector as a matrix
        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])

        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest),1])

        best_parameters = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [None])
    
            best_parameters[learnername] = cross_validate(
                10, Xtrain, Ytrain, Learner, params)

            # # stratified kfold test            
            # best_parameters[learnername] = stratifiedKfold(
            #     10, Xtrain, Ytrain, Learner, params)

        for learnername, Learner in classalgs.items():
            params = best_parameters[learnername]
            learner = Learner(params)
            learner.learn(Xtrain, Ytrain)
            predictions = learner.predict(Xtest)
            error = geterror(Ytest, predictions)
            errors[learnername][r] = error

    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        stderror = np.std(errors[learnername])/math.sqrt(numruns)
        print('Best parameters for ' + learnername + ': ' + str(best_parameters[learnername]))
        print('Average error for ' + learnername + ': ' + str(aveerror))
        print('Standard error for ' + learnername + ': ' + str(stderror))
