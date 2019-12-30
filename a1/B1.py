# Yuhan Ye 1463504 CMPUT 566
# code for Assignment1 Bonus d) print avg distance of samples of different dimensions

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


if __name__ == '__main__':
    dim = 2
    numsamples = 1000
    
 
    print("Running with dim = " + str(dim), \
        " and numsamples = " + str(numsamples))
        
    
    mu = np.zeros(dim)
    sigma = np.identity(dim)
    p1 = np.array(np.random.multivariate_normal(mu, sigma, numsamples).T)
    dist = np.sqrt(np.sum((p1)**2, axis=0))
        
    result = 0
    for i in dist:
        result = result + i
    result=result/numsamples
    print(result)


