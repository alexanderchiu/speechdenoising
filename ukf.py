from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def sigma(x,s,P):
    '''
    Parameters
        x - reference
        s - scaling factor
        P -  covariance

    Return

    sigma - sigma points
    '''
    n, d = x.shape
    L = n * d

    A = s*np.linalg.cholesky(P).T
    Y = np.tile(np.ravel(x),(L,1)).T
    X = np.hstack((x,Y+A, Y-A))

    return X

    def utrans(f,sigma,Wm,Wc,n,R):

    n, d = x.shape
    L = n * d

    A = s*np.linalg.cholesky(P).T
    Y = np.tile(np.ravel(x),(L,1)).T
    X = np.hstack((x,Y+A, Y-A))

    return X
def ukf(f, x, P, H, y, Q, R):
    '''
    Parameters
        f  - state model
        x - priori estimate of state
        P - priori estimate of state covariance
        H - measurement model
        y - Observation
        Q - Process noise covariance
        R - Measurement noise covariance

    Return
    '''
    # Parameters
    alpha = 0.001
    beta = 2
    kappa = 0

    n, d = x.shape
    L = n * d

    n, d = y.shape
    m = n * d

    lmda = alpha ** 2 * (L + kappa) - L
    c = L + lmda

    Wm = np.zeros(2*L)
    Wc = np.zeros(2*L)
    Wm[0] = lmda/c
    Wm[0] = lmda/c + (1 - alpha**2 +beta)

    s = np.sqrt(c)

    sigma = sigma(x,s,P)



x = np.matrix([[1],[1]])

s = 0.0017321
P =  np.array([[1.9902e-02,  5.5982e-06],[5.5982e-06,  2.1972e-02]])
X = sigma(x,s,P)



plt.show()