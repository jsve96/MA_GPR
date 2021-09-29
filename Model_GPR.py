import numpy as np
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from numpy.linalg import inv
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, cm

#Prior functions
class Prior_function(type):
    def __repr__(cls):
        return 'prior_function'
    
class gamma(metaclass=Prior_function):
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.typ = 'Gamma'
        self.bounds = (1e-5,None)
    def evaluate(self,x):
        return self.b**self.a/math.gamma(self.a)*x**(self.a-1)*np.exp(-self.b*x)
    def log_p(self,x):
        return np.log(self.evaluate(x))
    
class normal:
    def __init__(self,mu,s):
        self.mu = mu
        self.s = s
        self.typ = 'Normal'
        self.bounds = (None,None)
    def evaluate(self,x):
        return 1/np.sqrt(2*np.pi*self.s**2)*np.exp(-0.5*(x-self.mu)**2/self.s**2)
    def log_p(self,x):
        return np.log(self.evaluate(x))
    
#kernel
class RBF_2:
    def __init__(self, l=1.0,s=1.0):
        self.l = l
        self.s = s
    def evaluate(self,X,Xs=None,grad=False):
        if type(Xs) == type(None):
            Xs=X
        dist_norm = np.sum(X**2, 1).reshape(-1, 1) + np.sum(Xs**2, 1) - 2 * X@Xs.T
        #if str(type(self.l))=='prior_function':
            #self.l
        if grad:
            return [self.s**2*np.exp(-0.5*dist_norm/self.l)*dist_norm/self.l**3,2*self.s*np.exp(-0.5*dist_norm/self.l)]
        return self.s**2*np.exp(-0.5*dist_norm/self.l**2)
    

    
    
class Model:
    def __init__(self, name):
        self.name = name
        self.prior = {}
        self.cov = None
        self.mean = None
        self.theta = None
        self.theta_dict = None
        self.training_loss =[]
        self.res = None
        self.theta_df = None
        
    
    def add_prior(self,name,prior):
        if name not in self.prior:
            self.prior[name]=prior
        else:
            raise ValueError("variable name already in dict")
    def add_cov(self, Kernel):
        self.cov = Kernel
    
    def add_mean(self,mean):
        self.mean = mean
        
    def build_gp(self,theta,X_train,Y_train):
        K = self.cov(theta[0],theta[1]).evaluate(X_train) + theta[2]**2*np.eye(X_train.shape[0])
        if self.mean is None:
            mu = np.zeros(X_train.shape[0])
        else:
            mu = self.mean(*theta[3:]).evaluate(X_train)
        return K,mu
    
    def func_obj(self,X_train,Y_train):
        Y_train = Y_train.ravel()
        def log_p(theta):
            self.theta = theta
            if len(theta)!= len(self.prior):
                raise ValueError("add or remove prior or check if theta matches the number of optimization variables")
            
            self.theta_dict = {k: i for k,i in zip(self.prior.keys(),self.theta)}
            if isinstance(self.theta_df,type(None)):
                self.theta_df = pd.DataFrame(self.theta_dict, index=[0])
            else:
                self.theta_df =self.theta_df.append(self.theta_dict, ignore_index = True)
            if self.cov == None:
                raise ValueError("need to specify covariance function")
                
            
            K, mu = self.build_gp(theta,X_train,Y_train)
            K = K + 1e-5*np.eye(Y_train.shape[0])
            objective =0.5 * np.log(det(K)) + 0.5 * (Y_train-mu).dot(inv(K).dot(Y_train-mu)) + \
                       0.5 * len(X_train) * np.log(2*np.pi)
            for name,x in zip(self.prior.keys(),theta):
                objective-= self.prior[name].log_p(x)
            self.training_loss.append(objective)
            return objective
        return log_p
            
    def MAP(self,X_train,Y_train,theta):
        self.theta = theta
        #bounds noch anpassen
        bounds =[]
        for keys in self.prior:
            bounds.append(self.prior[keys].bounds)
        self.res = minimize(self.func_obj(X_train, Y_train), theta, 
               bounds=tuple(bounds),
               method='L-BFGS-B')
        #self.res = minimize(self.func_obj(X_train, Y_train), theta, method='BFGS')
        return self.res
    
    def posterior_dist(self,X_s, X_train, Y_train,return_vals = False):
        K = self.cov(*self.theta[:2]).evaluate(X_train) + self.theta[2]**2*np.eye(X_train.shape[0])
        K_s = self.cov(*self.theta[:2]).evaluate(X_train,X_s)
        K_ss = self.cov(*self.theta[:2]).evaluate(X_s) + 1e-8 * np.eye(X_s.shape[0])
        K_inv = inv(K)
        if self.mean is None:
            mu_s = np.zeros(X_s.shape[0])
            mu_train = np.zeros(X_train.shape[0])
        else:
            mu_s = self.mean(*theta[3:]).evaluate(X_s)
            mu_train = self.mean(*theta[3:]).evaluate(X_train)
        #mu_s = self.mean(*self.theta[3:]).evaluate(X_s)
        #mu_train = self.mean(*self.theta[3:]).evaluate(X_train)
        self.mu_s = mu_s + K_s.T.dot(K_inv).dot(Y_train-mu_train)
        self.cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        if return_vals:
            return self.mu_s,self.cov_s             
        
    def plot_post_dist(self,gx, gy, X_train, Y_train, title):
        ax = plt.gcf().add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(gx, gy, self.mu_s.reshape(gx.shape), cmap=cm.jet, linewidth=0, alpha=0.35, antialiased=True)
        ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.jet)
        ax.set_title(title)
        
    def summarize(self):
        print("model:{}\n self.prior :{}".format(self.name,self.prior))
        
        
        
class GPR(Model):
    def __init__(self,name):
        super().__init__(name)
        self.name = name
    def __enter__(self):
        #ttysetattr etc goes here before opening and returning the file object
        self.Model = Model(self.name)
        return self.Model
    def __exit__(self, type, value, traceback):
        #Exception handling here
        return 0
    
