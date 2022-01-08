import math
import numpy as np
import random

def make_checkerboard(size, n_samples, length, noise):
    size = math.sqrt(size)
    X = np.array([[random.uniform(0, length) for j in range(n_samples)] for i in range(2)])
    y = [0 if (math.ceil(x*size/length)+math.ceil(y*size/length))%2 == 0 else 1 for x, y in zip(X[0], X[1])]
    if noise:
        X = add_noise(X, length*0.5)
    return X.T, y

def add_noise(A, length):
    A = np.asarray([[i + random.uniform(-length/20, length/20) for i in array ] for array in A])
    return A

# https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html
def make_spirals(n_points, noise=False):
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))