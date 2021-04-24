# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 06:19:58 2021

@author: Yanis Zatout
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Multiply, Concatenate
from keras import backend as k
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

from tensorflow import squeeze
from tensorflow.math import multiply 

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def mult(a,b):
    return tf.cast(Multiply()([a,b]),dtype=tf.float64)

def concatenate(a,b,axis=0):
    return tf.cast(Concatenate(axis=axis)([a,b]),dtype=tf.float64)

def fitzhugh_nagumo(t, x, params = [-.3,1.4,20,0.23]): #(a, b, tau, I):
    a,b,tau,I=params
    return np.array([x[0] - x[0]**3 - x[1] + I,(x[0] - a - b * x[1])/tau])

def fitzhugh_nagumo_tf(t,x, params = [-.3,1.4,20,0.23]):
    a=tf.Variable(params[0],dtype=tf.float64)
    b=tf.Variable(params[2],dtype=tf.float64)
    tau=tf.Variable(params[3],dtype=tf.float64)
    I=tf.Variable(params[4],dtype=tf.float64)
    x=tf.transpose(x)
    return tf.Variable([x[0] - x[0]**3 - x[1] + I,(x[0] - a - b * x[1])/tau])

def f(t, y):  
    return t*(998*y[0] + 1998*y[1]), -999*y[0] - 1999*y[1]


#y0=[0.,0.]
y0=[2,-1]
t= np.linspace(0, 50, num=2000)
t2=np.linspace(0,2,11)
sol = solve_ivp(f, [t2.min(), t2.max()], y0, method='RK45', rtol=1e-5)
#sol = solve_ivp(fitzhugh_nagumo, [t.min(), t.max()], y0, method='RK45', rtol=1e-5)

def a(t,t0=0):
    return 1-np.exp(-(t-t0))

def b(t,t0=0):
  return np.exp(-(t-t0))

#@tf.function
def jacobian(func,input):
    with tf.GradientTape() as tape:
        tape.watch(input)
        preds=func(input)
    out=tf.squeeze(tape.batch_jacobian(preds,t))
    return out
            

@tf.function
def gradient(func,input,n_output,axis=0):
    """
    Returns gradient of all outputs w.r.t. your input
    """
    if axis == 1:
        return tf.squeeze([tf.gradients(func(input)[:,i],input)for i in range(n_output)])
    return tf.squeeze([tf.gradients(func(input)[i],input)for i in range(n_output)])

class lagaris_solver(tf.keras.Model):
    def __init__(self,layer_sizes,f = None,t=None,y0=None,a=None,b=None,activation=tf.nn.tanh,opt=None):
        super(lagaris_solver,self).__init__()
        self.nn =  Sequential()
        self.nn.add(Dense(layer_sizes[0], input_shape=(1,)))
        for layer_size in layer_sizes[1:-1]:
            self.nn.add(Dense(layer_size,activation = activation))
        self.nn.add(Dense(layer_sizes[-1]))
        
        if f is None:
            pass
        if t is None:
            pass
        if y0 is None:
            pass
        if a is None:
            def a(t):
                t0=tf.keras.backend.min(t)
                return 1-tf.math.exp(-(t-t0))
        if b is None:
            def b(t):
                t0=tf.keras.backend.min(t)
                return tf.math.exp(-(t-t0))
        self.f=f
        self.t=t
        self.y0=y0
        self.a=a
        self.b=b
        self.n_output=len(y0)
        self.n_points=len(t)
        self.OS=(self.n_points,self.n_output) #Output shape
        assert(self.n_output == layer_sizes[-1])
        self.hidden=layer_sizes[1:-1]
        if opt is None or "Adam":
            self.opt=keras.optimizers.Adam(learning_rate=1.e-3)
        
    def __call__(self,t):#Solution of Lagaris type
        y0 = tf.Variable(self.y0,dtype=tf.float64)
        return mult(self.a(t),self.nn(t)) + y0
    
    def __repr__(self):
        return self.__str__()
    
    @tf.function
    def gradcall(self,t):
        """
        obsolete function
        """
        gradcall=[tf.gradients(self.__call__(t)[:,i],t)  for i in range(self.n_output)]
        gradcall=tf.squeeze(gradcall)
        gradcall=tf.reshape(gradcall,self.OS)
        return gradcall
    
    def grad(self,t):
        computed_gradient=gradient(self.__call__, t, self.n_output,axis=1)
        return tf.reshape(computed_gradient, self.OS)
    
    def gradient(self,t):
        return self.grad(t)
    
    def jacobian(self,t=None):
        if t is None:
            t=self.t
        return jacobian(self.__call__,t)

    def __str__(self):
        return ('Neural ODE Solver \n'
                'Number of equations:       {} \n'
                'Initial condition y0:      {} \n'
                'Number of hidden units:   {} \n'
                'Number of training points: {} '
                .format(self.n_output, self.y0, self.hidden, np.size(self.t))
                )

    def summary(self):
        self.nn.summary()
        
    def __getitem__(self,key):
        return self.trainable_variables[key]

    
    
    def loss(self,t=None):
        
        if t is None:
            t=self.t
        
        nn_sol=self(t)
        grad_nn_sol=self.jacobian(t)
        actual_sol=self.f(t,nn_sol)
        diff=grad_nn_sol-actual_sol
        squared=tf.math.square(diff)
        scaled=self.b(t)*squared
        return tf.math.reduce_mean(scaled)
        
    
    def fit(self,t=None,iterations=1000,update_rate=100):
        if t is None:
            t=self.t
        error=[]
        for i in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(self.nn.trainable_variables)
                loss_value = self.loss(t)
            gradients = tape.gradient(loss_value,self.nn.trainable_variables)

            error.append(loss_value.numpy())
            grads_and_vars=zip(gradients, self.trainable_variables)
            self.opt.apply_gradients(grads_and_vars)
            
            if i % 100 == 0 or i == iterations:
                print("Iteration",i,"Loss:",loss_value.numpy())
            
        return error
        
    
    
    


def f_tf(t,y):

    a=mult(t,tf.Variable(998,dtype=tf.float64)*y[:,0]+tf.Variable(1998,dtype=tf.float64)*y[:,1])
    b=tf.reshape(tf.Variable(-999,dtype=tf.float64)*y[:,0] - tf.Variable(1999,dtype=tf.float64)*y[:,1],a.shape)
    return concatenate(a,b,axis=1)


t=tf.reshape(tf.linspace(0,2,11),(-1,1))
y0=[2,-1]
l=lagaris_solver([1,12,2],f_tf,t,y0)
l.loss(t)
#l.fit(t,1000)


plt.plot(l.t,l(t))
plt.plot(sol.t,sol.y.T)
