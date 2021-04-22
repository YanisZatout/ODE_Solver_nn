# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 06:19:58 2021

@author: Yanis Zatout
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as k
import tensorflow as tf
from tensorflow import squeeze

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def fitzhugh_nagumo(t, x, params = [-.3,1.4,20,0.23]): #(a, b, tau, I):
    a,b,tau,I=params
    return np.array([x[0] - x[0]**3 - x[1] + I,(x[0] - a - b * x[1])/tau])



y0=[0.,0.]
t= np.linspace(0, 50, num=2000)
#sol = solve_ivp(f, [t.min(), t.max()], y0, method='RK45', rtol=1e-5)
sol = solve_ivp(fitzhugh_nagumo, [t.min(), t.max()], y0, method='RK45', rtol=1e-5)

def a(t,t0=0):
    return 1-np.exp(-(t-t0))

def b(t,t0=0):
  return np.exp(-(t-t0))
layers=[1,32,32,2]
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
        self.n_output=len(f(self.t[0],self.y0))
        assert(self.n_output == layer_sizes[-1])
        self.hidden=layer_sizes[1:-1]
        if opt is None or "Adam":
            self.opt=keras.optimizers.Adam(learning_rate=1.e-3)
        
            
            
    def __call__(self,t):#Solution of Lagaris type
        return tf.keras.layers.Multiply()([tf.broadcast_to(self.a(t),(len(t),self.n_output)),self.nn(t)]) + self.y0
    def __repr__(self):
        return self.__str__()

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
        with tf.GradientTape() as tape:
            tape.watch(t)
            nn_sol=self(t)
        grad_nn_sol=tape.gradient(nn_sol,t)
        actual_sol=self.f(t,self.y0)
        
        return tf.math.reduce_mean(tf.math.square(grad_nn_sol-actual_sol))
        
    
    def fit(self,t=None,iterations=10000,update_rate=0.1):
        if t is None:
            t=self.t
        error=[]
        for i in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss_value = self.loss(t)
            gradients = tape.gradient(loss_value,self.trainable_variables)

            error.append(loss_value.numpy())
            grads_and_vars=zip(gradients, self.trainable_variables)
            self.opt.apply_gradients(grads_and_vars)
            
            if i+1 % iterations/(update_rate*100) == 0:
                print("Iteration",i,"Loss:",loss_value.numpy())
            
        return error
        
    
    
    
    
    
    
    
    
    
    
t=tf.reshape(tf.linspace(0,50,2000),(-1,1))
f=lagaris_solver(layers,fitzhugh_nagumo,t,y0)
error=f.fit()

plt.figure()
plt.plot(error)
plt.title("Error as a function of the number of cycles of optimisation")
plt.figure()
plt.plot(sol.t,sol.y.T)
plt.plot(f.t,f(t),"o")
plt.title("Comparaison of our class against a standard RK method")



