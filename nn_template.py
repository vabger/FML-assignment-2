import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Returns the ReLU value of the input x
def relu(x):
    return np.maximum(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x>0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
    return 1/(1 + np.exp(-x))

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    y = sigmoid(x)
    return y*(1-y)

## TODO 1c: Return the derivative of the tanh value of the input x
def tanh(x):
    return 2*sigmoid(2*x)-1

## TODO 1d: Return the derivative of the tanh value of the input x  
def tanh_derivative(x):
    return 1 - tanh(x)**2

# Mapping from string to function
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Given a list of activation functions, the following function returns
# the corresponding list of activation functions and their derivatives
def get_activation_functions(activations):  
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

class NN: 
    def __init__(self, input_dim, hidden_dims, activations=None):
        '''
        Parameters
        ----------
        input_dim : int
            size of the input layer.
        hidden_dims : LIST<int>
            List of positive integers where each integer corresponds to the number of neurons 
            in the hidden layers. The list excludes the number of neurons in the output layer.
            For this problem, we fix the output layer to have just 1 neuron.
        activations : LIST<string>, optional
            List of strings where each string corresponds to the activation function to be used 
            for all hidden layers. The list excludes the activation function for the output layer.
            For this problem, we fix the output layer to have the sigmoid activation function.
        ----------
        Returns : None
        ----------
        '''
        assert(len(hidden_dims) > 0)
        assert(activations == None or len(hidden_dims) == len(activations))
         
        # If activations is None, we use sigmoid activation for all layers
        if activations == None:
            self.activations = [sigmoid]*(len(hidden_dims)+1)
            self.activation_derivatives = [sigmoid_derivative]*(len(hidden_dims)+1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        ## TODO 2: Initialize weights and biases for all hidden and output layers
        ## Initialization can be done with random normal values, you are free to use
        ## any other initialization technique.
        self.weights = [np.random.normal(0, 1, (input_dim, hidden_dims[0]))] + \
               [np.random.normal(0, 1, (hidden_dims[i - 1], hidden_dims[i])) for i in range(1, len(hidden_dims))] + \
               [np.random.normal(0, 1, (hidden_dims[-1], 1))]


        self.biases = [np.random.normal(0, 1, (1, hidden_dims[i])) for i in range(len(hidden_dims))] +\
                       [np.random.normal(0, 1, (1, 1))]
        


    def forward(self, X):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        ----------
        Returns : output probabilities, numpy array of shape (N, 1) 
        ----------
        '''
        # Forward pass

        ## TODO 3a: Compute activations for all the nodes with the corresponding
        ## activation function of each layer applied to the hidden nodes        
        self.a = [X]
        self.z = []  
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = self.a[i] @ self.weights[i] + self.biases[i]  
            self.z.append(z)  
            a = self.activations[i](z) 
            self.a.append(a) 

        ## TODO 3b: Calculate the output probabilities of shape (N, 1) where N is number of examples
        self.o = self.a[-1] @ self.weights[-1] + self.biases[-1]
        output_probs = self.activations[-1](self.o)
        return output_probs

    def backward(self, X, y):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        y : target labels, numpy array of shape (N, 1) where N is the number of examples
        ----------
        Returns : gradients of weights and biases
        ----------
        '''
        # Backpropagation

        ## TODO 4a: Compute gradients for the output layer after computing derivative of 
        ## sigmoid-based binary cross-entropy loss
        ## Hint: When computing the derivative of the cross-entropy loss, don't forget to 
        ## divide the gradients by N (number of examples) 
        N = y.shape[0]
        yhat = self.forward(X)
        epsilon = 1e-8

        y = y.reshape(N,1)

        ye = yhat + epsilon
        dl_dyhat = -(y/(ye) - (1-y)/(1-ye)) # N x 1

        dz_output = dl_dyhat * self.activation_derivatives[-1](self.o)  # N x 1
        
        grad_out_weights = (self.a[-1].T @ dz_output) / N  # hidden_dims[-1] x 1
        grad_out_bias = np.sum(dz_output, axis=0, keepdims=True) / N  # 1 x 1
        
        ## TODO 4b: Next, compute gradients for all weights and biases for all layers
        ## Hint: Start from the output layer and move backwards to the first hidden layer
        self.grad_weights = []
        self.grad_biases = []

        self.grad_weights.insert(0,grad_out_weights)
        self.grad_biases.insert(0,grad_out_bias)


        dz = dz_output @ self.weights[-1].T
        for i in range(len(self.weights)-2,-1,-1):
            dz = dz * self.activation_derivatives[i](self.z[i])
            dw = (self.a[i].T @ dz )/N
            self.grad_weights.insert(0,dw)
            db = np.sum(dz, axis=0, keepdims=True)/N
            self.grad_biases.insert(0,db)

            if i > 0:
                dz = dz @ self.weights[i].T

        return self.grad_weights, self.grad_biases

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                gd_flag: 1 for Vanilla GD, 2 for GD with Exponential Decay, 3 for Momentum
                momentum: Momentum coefficient, used when gd_flag is 3.
                decay_constant: Decay constant for exponential learning rate decay, used when gd_flag is 2.
            epoch: Current epoch number
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        ### Calculate updated weights using methods as indicated by gd_flag
        updated_W = [None] * len(weights)
        updated_B = [None] * len(biases)

        ## TODO 5a: Variant 1(gd_flag = 1): Vanilla GD with Static Learning Rate
        ## Use the hyperparameter learning_rate as the static learning rate
        if gd_flag == 1:
            for i in range(len(weights)):
                updated_W[i] = weights[i] - learning_rate * delta_weights[i]
                updated_B[i] = biases[i] - learning_rate * delta_biases[i]
        

        ## TODO 5b: Variant 2(gd_flag = 2): Vanilla GD with Exponential Learning Rate Decay
        ## Use the hyperparameter learning_rate as the initial learning rate
        ## Use the parameter epoch for t
        ## Use the hyperparameter decay_constant as the decay constant
        elif gd_flag == 2:
            for i in range(len(weights)):
                updated_W[i] = weights[i] - learning_rate * delta_weights[i]
                updated_B[i] = biases[i] - learning_rate * delta_biases[i]
                learning_rate = learning_rate * np.exp(-(decay_constant*epoch))

        ## TODO 5c: Variant 3(gd_flag = 3): GD with Momentum
        ## Use the hyperparameters learning_rate and momentum
        elif gd_flag == 3:
            if not hasattr(self, 'velocity_w'):
                self.velocity_w = 0
                self.velocity_b = 0
        
            for i in range(len(weights)):
                w = momentum * self.velocity_w + (1-momentum) * delta_weights[i]
                b = momentum * self.velocity_b + (1-momentum) * delta_biases[i]

                updated_W[i] = weights[i] - learning_rate * w
                updated_B[i] = biases[i] - learning_rate * b
            

        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                beta: Exponential decay rate for the first moment estimates.
                gamma: Exponential decay rate for the second moment estimates.
                eps: A small constant for numerical stability.
        '''
        learning_rate = optimizer_params['learning_rate']
        beta = optimizer_params['beta']
        gamma = optimizer_params['gamma']
        eps = optimizer_params['eps']       

        ## TODO 6:dW, db Return updated weights and biases for the hidden layer based on the update rules for Adam Optimizer
        v_w = 0
        v_b = 0
        for i in range(len(weights)):
                v_w = beta * v_w + (1-beta) * delta_weights[i]
                v_b = beta * v_b + (1-beta) * delta_biases[i]
                s

        return updated_W, updated_B

    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Divide X,y into batches
            X_batches = np.array_split(X_train, X_train.shape[0]//batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0]//batch_size)
            for X, y in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X)
                # Backpropagation and gradient descent weight updates
                dW, db = self.backward(X, y)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)

            # Compute the training accuracy and training loss
            train_preds = self.forward(X_train)
            train_loss = np.mean(-y_train*np.log(train_preds) - (1-y_train)*np.log(1-train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            train_losses.append(train_loss)

            # Compute the test accuracy and test loss
            test_preds = self.forward(X_eval)
            test_loss = np.mean(-y_eval*np.log(test_preds) - (1-y_eval)*np.log(1-test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)

        return train_losses, test_losses

    
    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses, optimizer, optimizer_params):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if optimizer == "bgd":
            plt.savefig(f"loss_bgd_ {optimizer_params['gd_flag']}.png")
        else:
            plt.savefig(f'loss_adam.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X_train.shape[1]
    X_train = X_train**2
    X_eval = X_eval**2
    hidden_dims = [4,2] # the last layer has just 1 neuron for classification
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid']
    optimizer = "bgd"
    optimizer_params = {
        'learning_rate': 0.1,
        'gd_flag': 1,
        'momentum': 0.99,
        'decay_constant': 0.2
    }
    
    # For Adam optimizer you can use the following
    # optimizer = "adam"
    # optimizer_params = {
    #     'learning_rate': 0.01,
    #     'beta1' : 0.9,
    #     'beta2' : 0.999,
    #     'eps' : 1e-8
    # }

     
    model = NN(input_dim, hidden_dims)
    train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
    test_preds = model.forward(X_eval)

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)
