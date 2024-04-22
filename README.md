# NeuralNetwork
Implementing Neural network from scratch

# Implementation
In the following project we tried to implement a Multi layered and customisable Feed-Forward Artificial Neural Network (ANN) and a Particle Swarm Optimisation evolutionary algorithm (PSO) from the scratch. With the intent of adjusting the weights and biases of the Neural network with the evolutionary algorithm by the end of the project.
We approached this project in a step-by-step implementation process. During the project we were able to generate the following things from the scratch.
1.	A Multi layered and customizable feed-forward Artificial Neural Network (ANN).
2.	A Simple Particle Swarm Optimisation evolutionary algorithm
3.	Combining the above two to make a Multi layered Neural Network that trains with the PSO.

# Implementation and design of (ANN)
For the project we have implemented a multi-layered feed-forward configurable neural network with appropriate error handling for all layers, considering the popular test cases. Our neural network can take in different types of activation functions and calculate various model evaluation metrics which include error calculations as well. It also has a function to interact with PSO to receive the optimised weights using which the neural network gets updated to give more accurate predictions. As to retain customisability was one of the primary aims. Hence the user of this code can pass values for nearly all the hyper parameters of the ANN. 

# Implementation and design of (PSO)
We have implemented a basic version of PSO that has been customized to work with our multi-layered feed-forward neural network. PSO comes with a few nuances, namely calculating the total dimensions based on the neural network configurations. It does not have static hyperparameters but some of them do have a default value. Our fitness function is specially defined for adjusting weights and optimising them for the neural network. It also supports different type of error rectification functions based on the neural network configuration too because we wanted to retain customisability of the PSO as much as possible. So, the user of this code can pass values for nearly all the hyper parameters of the PSO with exception of number of dimensions.
Combining ANN with PSO
While combining the two algorithms to run with each other we observed that some modifications could be made with both algorithms to reduce the error that could occur when they run in their basic form. Such as we removed the input of number of dimensions from PSO and let the PSO figure it out based on all the layer configurations in the ANN. Another example would be a function in the Neural network class that can update its weights and biases with all the PSO optimised weights and biases, and final example would be the objective/fitness function of the PSO which is fully revamped to adjust neural network biases and weight according to the error rectification function of the neural network itself.
