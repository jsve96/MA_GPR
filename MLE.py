import numpy as np
from scipy.optimize import minimize
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, cm



def RBF(X1,X2,l,s):
    dist_norm = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1@X2.T
    return s**2*np.exp(-0.5*dist_norm/l**2)

def MLE(X_train,Y_train):
    Y_train = Y_train.ravel()
    def log_p(theta):
        K = RBF(X_train,X_train, theta[0], theta[1]) + theta[2]**2*np.eye(len(X_train)) 
        L = cholesky(K)
        A1 = solve_triangular(L, Y_train, lower=True)
        A2 = solve_triangular(L.T, A1, lower=False)
        objective =  np.sum(np.log(np.diagonal(L))) + 0.5 * Y_train.dot(A2) + 0.5*len(X_train) * np.log(2*np.pi)
        return objective
    return log_p

def plot_MLE_posterior_dist(gx,gy,X_train,Y_train,X_pred,theta,title):
    K = RBF(X_train,X_train, theta[0], theta[1]) + theta[2]**2*np.eye(len(X_train)) 
    L = cholesky(K)
    A1 = solve_triangular(L, Y_train, lower=True)
    A2 = solve_triangular(L.T, A1, lower=False)
    K_s = RBF(X_train,X_pred, theta[0], theta[1])
    mu = K_s.T@A2
              
    ax = plt.gcf().add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.jet, linewidth=0, alpha=0.35, antialiased=True)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.jet)
    ax.set_title(title)
              
          