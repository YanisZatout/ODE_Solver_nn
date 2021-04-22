# ODE_Solver_nn
This is a quick notebook I did to try and experiment with the Lagaris method for resolving ODEs with neural networks. I also added some techniques used in Flamant et al to speed up the learning rate. 

Every comment is made in French but a translation can be made if needed.
https://www.overleaf.com/read/kfdjbnjjpfkn
Here is the paper I wrote explaining the technique used.

References and inspirations are at the end of the notebook.


# Lagaris et al implementation in TensorFlow:

It's the same kind of implementation as in the notebook. I just made it in TensorFlow to try to solve the Fitzhugh-Nagumo model of ODEs. As show in the notebook, methods like Adam need many iterations to get to a satisfactory result. I didn't implement the BFGS method yet as it's not as easy to use as with scipy optimize minimize.
