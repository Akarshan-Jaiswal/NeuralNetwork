# NeuralNetwork
Implementing Neural network with different ways to adjust weights from scratch.

# Implementation
In the following project we tried to implement a Multi layered and customisable Feed-Forward Artificial Neural Network (ANN) with Backward Propogation and a Particle Swarm Optimisation evolutionary algorithm (PSO) from the scratch. With the intent of adjusting the weights and biases of the Neural network with either the evolutionary algorithm or backward propogation by the end of the project.
We approached this project in a step-by-step implementation process. During the project we were able to generate the following things from the scratch.
1.	A Multi layered and customizable feed-forward Artificial Neural Network (ANN).
2.	A Simple Particle Swarm Optimisation evolutionary algorithm
3.	Combining the above two to make a Multi layered Neural Network that trains with the PSO.

# Implementation and design of (ANN)
For the project we have implemented a multi-layered feed-forward configurable neural network with appropriate error handling for all layers, considering the popular test cases. Our neural network can take in different types of activation functions and calculate various model evaluation metrics which include error calculations as well. It also has a function to interact with backward propogation or PSO to receive the optimised weights using which the neural network gets updated to give more accurate predictions. As to retain customisability was one of the primary aims. Hence the user of this code can pass values for nearly all the hyper parameters of the ANN. 

# Implementation and design of (PSO)
We have implemented a basic version of PSO that has been customized to work with our multi-layered feed-forward neural network. PSO comes with a few nuances, namely calculating the total dimensions based on the neural network configurations. It does not have static hyperparameters but some of them do have a default value. Our fitness function is specially defined for adjusting weights and optimising them for the neural network. It also supports different type of error rectification functions based on the neural network configuration too because we wanted to retain customisability of the PSO as much as possible. So, the user of this code can pass values for nearly all the hyper parameters of the PSO with exception of number of dimensions.
Combining ANN with PSO
While combining the two algorithms to run with each other we observed that some modifications could be made with both algorithms to reduce the error that could occur when they run in their basic form. Such as we removed the input of number of dimensions from PSO and let the PSO figure it out based on all the layer configurations in the ANN. Another example would be a function in the Neural network class that can update its weights and biases with all the PSO optimised weights and biases, and final example would be the objective/fitness function of the PSO which is fully revamped to adjust neural network biases and weight according to the error rectification function of the neural network itself.

# Code Breakdown
## Neural network with backward propogation

This implementation now includes:

1) Activation Functions: Sigmoid, ReLU, and Tanh with their derivatives.
2) Loss Functions: Mean Squared Error and Cross-Entropy with their derivatives.
3) Regularization: L1 and L2 regularization with their derivatives.
4) Optimizers: SGD, Adam, and AdaMax.
6) Callbacks: EarlyStopping for stopping training when the loss doesn't improve.
7) History: Tracking and printing the loss at the end of each epoch.
8) Evaluation metrics : Evaluating model against several parameters like Accuracy, Mean Square Error, Log Loss, Precision and more.

The train method accepts callbacks to integrate these features, and the optimizer is used to update weights and biases during training.

Library Imports
```bash
import numpy as np
```

### ActivationFunction: 
This class provides static methods for commonly used activation functions (sigmoid, relu, tanh) and their derivatives. These functions introduce non-linearity, allowing the network to learn complex relationships in the data.
```bash
'''
ActivationFunction: 
This class provides static methods for commonly used activation functions in neural networks, along with their derivatives. 
These functions introduce non-linearity into the network, allowing it to learn complex relationships in the data.
'''

class ActivationFunction:
    # sigmoid: Applies the sigmoid function (squashes values between 0 and 1).
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # sigmoid_derivative: Calculates the derivative of the sigmoid function.
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    # relu: Applies the rectified linear unit (ReLU) function (outputs the input directly if positive, otherwise 0).
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    # relu_derivative: Calculates the derivative of the ReLU function.
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    # tanh: Applies the hyperbolic tangent function (squashes values between -1 and 1).
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    # tanh_derivative: Calculates the derivative of the hyperbolic tangent function.
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
```

### LossFunction:
This class provides static methods for calculating loss functions (mean squared error, cross-entropy, log loss) used to measure how well the network's predictions match the targets.
Includes clipping for cross-entropy to prevent division by zero.
```bash
'''
LossFunction: 
This class provides static methods for calculating loss functions, which measure how well the networks predictions match the actual targets.
'''

class LossFunction:
    # mean_squared_error: Calculates the mean squared error between predicted and true values.
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    # mean_squared_error_derivative: Calculates the derivative of the mean squared error.
    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    
    # cross_entropy: Calculates the cross-entropy loss (commonly used for classification problems).
    @staticmethod
    def cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # cross_entropy_derivative: Calculates the derivative of the cross-entropy loss.
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    # log_loss: Calculates the log loss between predicted and true values.
    @staticmethod
    def log_loss(y_true, y_pred):
        return -np.mean(y_true * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)) + (1 - y_true) * np.log(1 - np.clip(y_pred, 1e-15, 1 - 1e-15)))
```

### Regularization:
This class provides static methods for regularization techniques (L1, L2) used to prevent overfitting by penalizing large weights.
```bash
'''
Regularization: 
This class provides static methods for regularization techniques, which help prevent overfitting by penalizing large weights.
'''

class Regularization:
    # l1: Calculates the L1 norm of the weights (sum of absolute values).
    @staticmethod
    def l1(weights, alpha):
        return alpha * np.sum(np.abs(weights))
    
    # l1_derivative: Calculates the derivative of the L1 norm.
    @staticmethod
    def l1_derivative(weights, alpha):
        return alpha * np.sign(weights)
    
    # l2: Calculates the L2 norm of the weights (sum of squares).
    @staticmethod
    def l2(weights, alpha):
        return alpha / 2 * np.sum(np.square(weights))
    
    # l2_derivative: Calculates the derivative of the L2 norm.
    @staticmethod
    def l2_derivative(weights, alpha):
        return alpha * weights
```

### Optimizer:
This is an abstract class that defines the interface for updating the network's weights during training. Subclasses implement specific optimization algorithms (SGD, Adam).
```bash
'''
Optimizer: 
This is an abstract class that defines the interface for updating the networks weights during training. 
Subclasses implement specific optimization algorithms.
'''
class Optimizer:
    # update: This method is not implemented in the abstract class but should be defined by subclasses to update weights based on gradients and learning rate.
    def update(self, weights, gradients, learning_rate):
        raise NotImplementedError
```

### SGD (Stochastic Gradient Descent):
This class implements the SGD optimizer, which updates weights based on the gradients of a single data point at a time.
```bash
'''
SGD (Stochastic Gradient Descent):
This class implements the SGD optimizer, which updates weights based on the gradients of a single data point at a time.
'''
class SGD(Optimizer):
    # update: Updates weights by subtracting the learning rate times the gradients.
    def update(self, weights, gradients, learning_rate):
        return weights - learning_rate * gradients
```
### Adam (Adaptive Moment Estimation): 
This class implements the Adam optimizer, a more advanced algorithm that adapts the learning rate for each weight based on past gradients.

Uses momentum and adaptive learning rates with proper initialization.
```bash
'''
Adam (Adaptive Moment Estimation):
This class implements the Adam optimizer, a more advanced algorithm that adapts the learning rate for each weight based on past gradients.
'''
class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    # update: Updates weights using the Adam algorithm with momentum and adaptive learning rates.
    def update(self, weights, gradients, learning_rate, layer_id):
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(weights)
            self.v[layer_id] = np.zeros_like(weights)
        
        self.t += 1
        self.m[layer_id] = self.beta1 * self.m[layer_id] + (1 - self.beta1) * gradients
        self.v[layer_id] = self.beta2 * self.v[layer_id] + (1 - self.beta2) * np.square(gradients)

        m_hat = self.m[layer_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer_id] / (1 - self.beta2 ** self.t)

        return weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### EarlyStopping:
This class implements a callback for early stopping during training.
Monitors validation loss and stops training if it doesn't improve for a certain number of epochs.
```bash
'''
EarlyStopping: 
This class implements a callback for early stopping during training. 
It monitors the validation loss and stops training if the loss doesnt improve for a certain number of epochs (iterations).
'''
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait = 0
        self.stopped_epoch = 0

    # __call__: Checks if the validation loss has stopped improving and returns True for early stopping.
    def __call__(self, current_loss, epoch):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False
```

### History:
This class keeps track of the loss values during training for monitoring and visualization purposes.
```bash
'''
History:
This class keeps track of the loss values during training for monitoring and visualization purposes.
'''
class History:
    def __init__(self):
        self.losses = []

    # on_epoch_end: Appends the loss value for the current epoch to the internal list.
    def on_epoch_end(self, epoch, loss):
        self.losses.append(loss)
        print(f'Epoch {epoch + 1}: loss = {loss}')
```

### NeuralNetwork:

This is the main class that builds and trains the neural network.

- Provides methods for adding layers, setting loss function, optimizer, and performing forward and backward propagation.
- Includes regularization and early stopping functionalities.
- Evaluation Metrics: This method seems like it could be placed in a separate class for better organization. It calculates various metrics like accuracy, precision, recall, etc.
```bash
'''
NeuralNetwork: 
This is the main class that builds trains and evaluates the neural network.
'''
class NeuralNetwork:
    # __init__: Initializes the network with empty layers, loss function, optimizer, etc.
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.loss_derivative = None
        self.regularization = None
        self.regularization_param = 0
        self.optimizer = None

    # add_layer: Adds a new layer to the network with a specified number of inputs, neurons, and activation function.
    def add_layer(self, num_inputs, num_neurons, activation_function='sigmoid'):
        layer = {
            'weights': np.random.randn(num_inputs, num_neurons) * 0.1,
            'biases': np.zeros((1, num_neurons)),
            'activation_function': getattr(ActivationFunction, activation_function),
            'activation_derivative': getattr(ActivationFunction, activation_function + '_derivative')
        }
        self.layers.append(layer)

    # set_loss_function: Sets the loss function used for calculating the error between predictions and targets.
    def set_loss_function(self, loss_function):
        self.loss_function = getattr(LossFunction, loss_function)
        self.loss_derivative = getattr(LossFunction, loss_function + '_derivative')

    # set_regularization: Sets the regularization technique and its parameter (alpha) used to prevent overfitting.
    def set_regularization(self, regularization, alpha):
        self.regularization = getattr(Regularization, regularization)
        self.regularization_derivative = getattr(Regularization, regularization + '_derivative')
        self.regularization_param = alpha

    # set_optimizer: Sets the optimizer algorithm used for updating the network's weights during training.
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # forward_propagation: Propagates the input data through the network's layers, applying activation functions at each step.
    def forward_propagation(self, X):
        for layer in self.layers:
            Z = np.dot(X, layer['weights']) + layer['biases']
            A = layer['activation_function'](Z)
            layer['Z'] = Z
            layer['A'] = A
            X = A
        return X
    
    # backward_propagation: Calculates the gradients of the loss function with respect to the weights and biases, used for updating them during training.
    def backward_propagation(self, X, y, learning_rate):
        m = y.size
        dA = self.loss_derivative(y, self.layers[-1]['A'])
        
        for i in reversed(range(len(self.layers))):
            dZ = dA * self.layers[i]['activation_derivative'](self.layers[i]['A'])
            dW = np.dot(self.layers[i-1]['A'].T, dZ) if i > 0 else np.dot(X.T, dZ)
            dB = np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, self.layers[i]['weights'].T)
            
            if self.regularization:
                dW += self.regularization_derivative(self.layers[i]['weights'], self.regularization_param)
            
            if self.optimizer:
                self.layers[i]['weights'] = self.optimizer.update(self.layers[i]['weights'], dW, learning_rate, f'layer_{i}_weights')
                self.layers[i]['biases'] = self.optimizer.update(self.layers[i]['biases'], dB, learning_rate, f'layer_{i}_biases')
            else:
                self.layers[i]['weights'] -= learning_rate * dW
                self.layers[i]['biases'] -= learning_rate * dB
    
    # calculate_loss: Calculates the loss value based on the chosen loss function and any regularization penalty.
    def calculate_loss(self, y_true, y_pred):
        loss = self.loss_function(y_true, y_pred)
        if self.regularization:
            for layer in self.layers:
                loss += self.regularization(layer['weights'], self.regularization_param)
        return loss
    
    # train: Trains the network on a given dataset for a specified number of epochs (iterations) using the chosen optimizer and learning rate. 
    # It also supports callbacks for monitoring the training process.
    def train(self, X, y, epochs, learning_rate, callbacks=[]):
        history = History()
        for epoch in range(epochs):
            y_pred = self.forward_propagation(X)
            loss = self.calculate_loss(y, y_pred)
            self.backward_propagation(X, y, learning_rate)
            history.on_epoch_end(epoch, loss)
            for callback in callbacks:
                if callback(loss, epoch):
                    print(f'Early stopping at epoch {epoch + 1}')
                    return history
        return history
    
    # predict: Uses the trained network to make predictions on a new set of input data.
    def predict(self, X):
        return self.forward_propagation(X)
    
    # evaluation_metrics: Calculate various model evaluation metrics which include error calculations as well.
    def evaluation_metrics(self,target_list,prediction_list):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for l1,l2 in zip(target_list, prediction_list):
            if (l1 == 1 and  np.round(l2) == 1):
                true_positive = true_positive + 1
            elif (l1 == 0 and np.round(l2) == 0):
                true_negative = true_negative + 1
            elif (l1 == 1 and np.round(l2) == 0):
                false_negative = false_negative + 1
            elif (l1 == 0 and np.round(l2) == 1):
                false_positive = false_positive + 1
        result={}

        binary_predictions = np.round(prediction_list)
        correct_predictions = np.sum(binary_predictions == target_list)
        total_predictions = len(target_list)
        
        result["Accuracy"] = correct_predictions / total_predictions
        result["Mean Square Error"] = np.mean(np.square(target_list - predictions))
        result["Mean Absolute Error"] = np.mean(np.abs(target_list - prediction_list))
        LL = -np.mean(target_list * np.log(np.clip(prediction_list, 1e-15, 1 - 1e-15)) + (1 - target_list) * np.log(1 - np.clip(prediction_list, 1e-15, 1 - 1e-15)))
        result["Log Loss"] = LL
        
        result["True Positives"] = true_positive
        result["True Negatives"] = true_negative
        result["False Positives"] = false_positive
        result["False Negatives"] = false_negative

        precision=0
        recall=0
        if(true_positive + false_positive)!=0:
            precision = true_positive/(true_positive + false_positive)
            recall = true_positive/(true_positive + false_negative)
        result["Precision"] = precision
        result["Recall"] = recall
        result["Specificity"] = 0
        if (true_negative + false_positive)!=0:
            result["Specificity"] = true_negative/(true_negative + false_positive)
        result["Negative Predictive Value"] = 0
        if (true_negative + false_negative)!=0:
            result["Negative Predictive Value"] = true_negative/(true_negative + false_negative)
        result["F-Measure"] = 0
        if (precision + recall)!=0:
            result["F-Measure"] = (2* precision * recall)/(precision + recall) #F1 score
        return result
```

# Work in Progress