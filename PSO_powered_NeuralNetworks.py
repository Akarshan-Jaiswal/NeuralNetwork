#Final
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_hidden="sigmoid", activation_output="sigmoid",error_calculation="Log Loss"):
        
        if input_size<1:
            raise ValueError("Invalid amount of neuron in input layer")
        if output_size<1:
            raise ValueError("Invalid amount of neuron in output layer")
        if len(hidden_size)<1:
            raise ValueError("Hidden layer is empty")
        if min(hidden_size)<1:
            raise ValueError(f"Invalid amount of neuron in hidden layer configuration:{min(hidden_size)}")
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size[0])
        self.biases_hidden = np.random.rand(1, hidden_size[0])
        self.weights_layer_hidden = []
        self.biases_layer_hidden = []
        for layer in range(0,len(hidden_size)-1):
            self.weights_layer_hidden.append(np.random.rand(hidden_size[layer], hidden_size[layer+1]))
            self.biases_layer_hidden.append(np.random.rand(1, hidden_size[layer+1]))
        self.weights_hidden_output = np.random.rand(hidden_size[-1], output_size)
        self.biases_output = np.random.rand(1, output_size)

        self.configuration={}
        self.configuration["input size"]=input_size
        self.configuration["hidden size"]=hidden_size
        self.configuration["output size"]=output_size

        if error_calculation == 'Log Loss':
            self.error_calculation='Log Loss'
        elif error_calculation == 'Mean square Error':
            self.error_calculation='Mean square Error'
        elif error_calculation == 'Mean Absolute Error':
            self.error_calculation= 'Mean Absolute Error'
        else:
            raise ValueError("Invalid error calculation metrics")

        # Set activation functions
        if activation_hidden == 'sigmoid':
            self.activation_hidden = self.sigmoid
        elif activation_hidden == 'relu':
            self.activation_hidden = self.relu
        elif activation_hidden == 'tanh':
            self.activation_hidden = self.tanh
        else:
            raise ValueError("Invalid activation function")
        
        if activation_output == 'sigmoid':
            self.activation_output = self.sigmoid
        elif activation_output == 'relu':
            self.activation_output = self.relu
        elif activation_output == 'tanh':
            self.activation_output = self.tanh
        else:
            raise ValueError("Invalid activation function")

    def forward(self, X):

        # Forward pass through the network
        self.hidden_output = np.dot(X, self.weights_input_hidden) + self.biases_hidden
        self.hidden_activation = self.activation_hidden(self.hidden_output)
        self.layer_output = []
        self.layer_activation = []
        is_hidden=True
        for layer in range(0,len(self.configuration["hidden size"])-1):
            temp_layer=0
            temp_activation=0
            if is_hidden:
                temp_layer=np.dot(self.hidden_activation, self.weights_layer_hidden[layer]) + self.biases_layer_hidden[layer]
                temp_activation=self.activation_hidden(temp_layer)
            else:
                layer_hidden=np.dot(self.weights_layer_hidden[layer-1], self.weights_layer_hidden[layer]) + self.biases_layer_hidden[layer]
                layer_hidden=self.activation_hidden(layer_hidden)
                temp_layer=np.dot(layer_hidden,self.weights_layer_hidden[layer]) + self.biases_layer_hidden[layer]
                temp_activation=self.activation_hidden(temp_layer)
                is_hidden=False
            self.layer_output.append(temp_layer)
            self.layer_activation.append(temp_activation)
        self.final_output = np.dot(self.layer_activation[-1], self.weights_hidden_output) + self.biases_output
        return self.activation_output(self.final_output)
    
    # Activation functions
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def relu(self,x):
        return np.maximum(0, x)

    def tanh(self,x):
        return np.tanh(x)
    
    def load_data(self,training_inputs,training_labels,layer_generation=True):
        self.training_data=training_inputs
        self.training_labels=training_labels

        if layer_generation:
            self.configuration["hidden size"][0]=max(self.configuration["hidden size"])
            self.weights_input_hidden = np.random.rand(self.configuration["input size"], self.configuration["hidden size"][0])
            self.biases_hidden = np.random.rand(1, self.configuration["hidden size"][0])
            self.weights_layer_hidden = []
            self.biases_layer_hidden = []
            for layer in range(0,len(self.configuration["hidden size"])-1):
                self.weights_layer_hidden.append(np.random.rand(self.configuration["hidden size"][0], self.configuration["hidden size"][0]))
                self.biases_layer_hidden.append(np.random.rand(1, self.configuration["hidden size"][0]))
            self.weights_hidden_output = np.random.rand(self.configuration["hidden size"][-1], self.configuration["output size"])
            self.biases_output = np.random.rand(1, self.configuration["output size"])

            hiddensize=len(self.configuration["hidden size"])
            pivotvalue=self.configuration["hidden size"][0]
            self.configuration["hidden size"]=[pivotvalue for _ in range(hiddensize)]
            self.configuration["hidden size"].append(1)
    
    def flush_training_data(self):
        self.training_data=None
        self.training_labels=None

    def set_PSO_optimised_weights(self,optimised_weigths):

        input_size=self.configuration["input size"]
        hidden_size=self.configuration["hidden size"]
        output_size=self.configuration["output size"]
        best_weights=optimised_weigths

        num_weights_input_hidden = input_size * hidden_size[0]
        num_biases_hidden = hidden_size[0]
        best_weights_input_hidden = np.reshape(best_weights[:num_weights_input_hidden], (input_size, hidden_size[0]))
        best_biases_hidden = np.reshape(best_weights[num_weights_input_hidden:num_weights_input_hidden + num_biases_hidden],
                                        (1, hidden_size[0]))
        num_weights_layer=[]
        num_biases_layer=[]
        best_weights_layer=[]
        best_biases_layer=[]
        for layer in range(0,len(hidden_size)-1):
            temp_weight_layer=hidden_size[layer]*hidden_size[layer+1]
            num_weights_layer.append(temp_weight_layer)
            num_biases_layer.append(hidden_size[layer+1])
            if layer == 0:
                layer_weight=num_weights_input_hidden + num_biases_hidden
                layer_weight_with_temp=num_weights_input_hidden + num_biases_hidden + temp_weight_layer
                best_weights_layer.append(np.reshape(best_weights[layer_weight:layer_weight_with_temp],(hidden_size[layer], hidden_size[layer+1])))
                best_biases_layer.append(np.reshape(best_weights[layer_weight_with_temp:layer_weight_with_temp+hidden_size[layer+1]],(1, hidden_size[layer+1])))
            else:
                layer_weight=num_weights_layer[layer] + num_biases_layer[layer]
                layer_weight_with_temp=num_weights_layer[layer] + num_biases_layer[layer] + temp_weight_layer
                best_weights_layer.append(np.reshape(best_weights[layer_weight:layer_weight_with_temp],(hidden_size[layer], hidden_size[layer+1])))
                best_biases_layer.append(np.reshape(best_weights[layer_weight_with_temp:layer_weight_with_temp+hidden_size[layer+1]],(1, hidden_size[layer+1])))
        num_weights_hidden_output = hidden_size[-1] * output_size

        layer_weight=num_weights_layer[-1] + num_biases_layer[-1]
        layer_weight_with_temp=num_weights_layer[-1] + num_biases_layer[-1] + num_weights_hidden_output
        best_weights_hidden_output = np.reshape(best_weights[layer_weight:layer_weight_with_temp],(hidden_size[-1], output_size))
        best_biases_output = np.reshape(best_weights[layer_weight_with_temp:layer_weight_with_temp+output_size],(1, output_size))

        self.weights_input_hidden = best_weights_input_hidden
        self.biases_hidden = best_biases_hidden
        self.weights_layer_hidden = best_weights_layer
        self.biases_layer_hidden = best_biases_layer
        self.weights_hidden_output = best_weights_hidden_output
        self.biases_output = best_biases_output

    def predict(self,value_list):
        return self.forward(value_list)
    
    #Evalution metrics
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

# Define the PSO algorithm
def particle_swarm_optimization(objective_function,p_neural_network, num_particles, max_iter, inertia_weight=0.5, cognitive_weight=2.0,social_weight=2.0):
    # Initialize particles' positions and velocities
    input_size=p_neural_network.configuration["input size"]
    hidden_size=p_neural_network.configuration["hidden size"]
    output_size=p_neural_network.configuration["output size"]
    num_dimensions = (input_size * hidden_size[0]) + hidden_size[0]
    for layer in range(0,len(hidden_size)-1):
        num_dimensions+=(hidden_size[layer] * hidden_size[layer+1]) + hidden_size[layer+1]
    num_dimensions+=(hidden_size[-1] * output_size) + output_size
    particles_position = np.random.rand(num_particles, num_dimensions)
    particles_velocity = np.random.rand(num_particles, num_dimensions)

    # Initialize personal best positions and fitness values
    personal_best_positions = particles_position.copy()
    personal_best_fitness = np.full(num_particles, np.inf)

    # Initialize global best position and fitness value
    global_best_position = np.zeros(num_dimensions)
    global_best_fitness = np.inf

    for iteration in range(max_iter):
        for i in range(num_particles):
            # Evaluate fitness
            fitness = objective_function(particles_position[i],p_neural_network)

            # Update personal best
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles_position[i]

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particles_position[i]

            # Update velocity and position
            inertia_weight = inertia_weight
            cognitive_weight = cognitive_weight
            social_weight = social_weight

            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = cognitive_weight * r1 * (personal_best_positions[i] - particles_position[i])
            social_component = social_weight * r2 * (global_best_position - particles_position[i])

            particles_velocity[i] = inertia_weight * particles_velocity[i] + cognitive_component + social_component
            particles_position[i] = particles_position[i] + particles_velocity[i]

    p_neural_network.flush_training_data()
    return global_best_position

# Define the objective function to minimize error
def fitness_function(weights,p_neural_network):
    # Reshape weights for the neural network
    input_size = p_neural_network.configuration["input size"]
    hidden_size = p_neural_network.configuration["hidden size"]
    output_size = p_neural_network.configuration["output size"]

    num_weights_input_hidden = input_size * hidden_size[0]
    num_biases_hidden = hidden_size[0]
    num_weights_hidden_output = hidden_size[-1] * output_size
    num_biases_output = output_size

    weights_input_hidden = np.reshape(weights[:num_weights_input_hidden], (input_size, hidden_size[0]))
    biases_hidden = np.reshape(weights[num_weights_input_hidden:num_weights_input_hidden + num_biases_hidden],
                                (1, hidden_size[0]))

    num_weights_layer=[]
    num_biases_layer=[]
    weights_layer=[]
    biases_layer=[]
    for layer in range(0,len(hidden_size)-1):
        temp_weight_layer=hidden_size[layer]*hidden_size[layer+1]
        num_weights_layer.append(temp_weight_layer)
        num_biases_layer.append(hidden_size[layer+1])
        if layer == 0:
            layer_weight=num_weights_input_hidden + num_biases_hidden
            layer_weight_with_temp=num_weights_input_hidden + num_biases_hidden + temp_weight_layer

            weights_layer.append(np.reshape(weights[layer_weight:layer_weight_with_temp],(hidden_size[layer], hidden_size[layer+1])))
            biases_layer.append(np.reshape(weights[layer_weight_with_temp:layer_weight_with_temp+hidden_size[layer+1]],(1,hidden_size[layer+1])))
        else:
            layer_weight=num_weights_layer[layer] + num_biases_layer[layer]
            layer_weight_with_temp=num_weights_layer[layer] + num_biases_layer[layer] + temp_weight_layer
            weights_layer.append(np.reshape(weights[layer_weight:layer_weight_with_temp],(hidden_size[layer], hidden_size[layer+1])))
            biases_layer.append(np.reshape(weights[layer_weight_with_temp:layer_weight_with_temp+hidden_size[layer+1]],(1,hidden_size[layer+1])))

    layer_weight=num_weights_layer[-1] + num_biases_layer[-1]
    layer_weight_with_output=num_weights_layer[-1] + num_biases_layer[-1] + num_weights_hidden_output
    weights_hidden_output = np.reshape(weights[layer_weight:layer_weight_with_output],(hidden_size[-1], output_size))
    biases_output = np.reshape(weights[layer_weight_with_output:layer_weight_with_output+output_size],
                                (1, output_size))

    # Set the weights and biases in the neural network
    p_neural_network.weights_input_hidden = weights_input_hidden
    p_neural_network.biases_hidden = biases_hidden
    p_neural_network.weights_layer_hidden = weights_layer
    p_neural_network.biases_layer_hidden = biases_layer
    p_neural_network.weights_hidden_output = weights_hidden_output
    p_neural_network.biases_output = biases_output
    
    # Forward pass through the neural network
    predictions = p_neural_network.forward(p_neural_network.training_data)
    labels=p_neural_network.training_labels

    # Calculate error
    error=np.mean(np.square(predictions - labels))
    if p_neural_network.error_calculation == 'Log Loss':
        error = -np.mean(labels * np.log(np.clip(predictions, 1e-15, 1 - 1e-15)) + (1 - labels) * np.log(1 - np.clip(predictions, 1e-15, 1 - 1e-15)))
    elif p_neural_network.error_calculation == 'Mean square Error':
        error = np.mean(np.square(predictions - labels))
    elif p_neural_network.error_calculation == 'Mean Absolute Error':
        error = np.mean(np.abs(labels - predictions))

    return error

# Load data from CSV file
#data = pd.read_csv('Bank_Data 1.csv')
data = pd.read_csv('Data/Bank_Data 1.csv')


# Assuming the last column is the target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the input, hidden, and output sizes
input_size = X_train.shape[1]
hidden_size = [5,4,3,9]
output_size = 1

# Create an instance of the neural network with ReLU activation for hidden layer and error as Log Loss
neural_network = NeuralNetwork(input_size, hidden_size, output_size, activation_hidden="relu",error_calculation="Log Loss")
neural_network.load_data(X_train,y_train)

# Set PSO parameters
num_particles = 15
max_iter = 100

# Run PSO to optimize neural network weights

best_weights = particle_swarm_optimization(fitness_function,neural_network, num_particles, max_iter)

# Set PSO optimised weights in the neural network

neural_network.set_PSO_optimised_weights(best_weights)

predictions = neural_network.predict(X_test)

evaluation=neural_network.evaluation_metrics(y_test,predictions)
print("Optimized Neural Network Predictions:")
print(evaluation)

cm=[[evaluation["True Negatives"],evaluation["False Positives"]],[evaluation["False Negatives"],evaluation["True Positives"]]]
print(cm)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="coolwarm")
plt.show()
