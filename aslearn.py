import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class Activation_Functions:
    """# Activation Function Library"""

    def linear(z:list, derivative:bool = False) -> list:
        if derivative:
            return np.ones(z.shape)
        return z

    def sigmoid(z:list, derivative:bool = False) -> list:
        if derivative:
            z = Activation_Functions.sigmoid(z)
            return z * (1-z)  
        z = np.clip(z, -250, 250) #make numericaly stable
        return 1/(1+np.exp(-z))

    def relu(z, derivative:bool=False):
        if derivative:
            z = np.where(z >= 0, 1, 0)
        return np.maximum(0, z)

    def leaky_relu(z, derivative:bool=False):
        if derivative:
            z = np.where(z >= 0, 1, 0.01)
            return z
        return np.where(z >= 0, 1*z, 0.01*z)

    def tanh(z, derivative:bool=False):
        if derivative:
            return 1.0 - np.tanh(z) ** 2
        return np.tanh(z)

    def softmax(z, derivative:bool=False):
        z = z - z.max() # make numericaly stable
        exps = np.exp(z)
        if derivative:
            return exps / np.sum(exps, axis=1, keepdims=True) * (1 - exps / np.sum(exps, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def factory(function:str):
        """# Activation Factory
        Factory that returns an activation function.
        
        ### Suported activation functions:
            - linear: used for linear separable problems
            - sigmoid: used for logistic regression
            - relu: most common for hidden layers in networks
            - leaky_relu: improved relu function
            - tanh: alternative for hidden layers in networks
            - softmax: all outputs sum to 1, used to generate probabilities
        
        ### Args:
            - function: name of activation function as a string

        ### Returns:
            - activation function

        ### Raises:
            - ValueError: if input is not a supported activation function
        """

        if function == 'linear':
            return Activation_Functions.linear
        elif function == 'logistic':
            return Activation_Functions.sigmoid
        elif function == 'sigmoid':
            return Activation_Functions.sigmoid
        elif function == 'relu':
            return Activation_Functions.relu
        elif function == 'leaky_relu':
            return Activation_Functions.leaky_relu
        elif function == 'tanh':
            return Activation_Functions.tanh
        elif function == 'softmax':
            return Activation_Functions.softmax
        else:
            raise ValueError('"{}" is not a supported function.'.format(function))

class Neural_Network:
    """# Neural Network
    Create a Neural Network with support for regression training as well as genetic algos. 
    ### Kwargs:
    - config: List definining the hidden layers and output specifications.
        example: 2 hidden ReLu layers with 10 softmax output:
            [[16, 'relu'],[16, 'relu'],[10, 'softmax']]
        example: 4 hidden ReLu layers with 2 softmax output:
            [[16, 'relu'],[32, 'relu'],[64, 'relu'],[8, 'relu'],[2, 'softmax']]
    - input: [Default None] Amount of nodes in the input layer, will automatically be set durring training wait back prop."""


    def __init__(self, config:list = [[16, 'leaky_relu'],[16, 'leaky_relu'],[10, 'softmax']], input_size:int=None):

                self.config = config
                self.thetas = []
                self.activations = []

                self.epochs = 0
                self.J_history = {'J':[]}
                self.accuracy_history = {'Train':[], 'Test':[]}

                if input_size:
                    X = np.zeros((1,input_size))
                    self.initialize(X)

    def initialize(self, X):

                m = len(X[0])

                input_config = [[m, 'input']]
                self.config = input_config + self.config

                n_layers = len(self.config)

                self.thetas = []
                self.activations = []

                for i in range(1, n_layers):
                    previous_layer_size = self.config[i-1][0]
                    current_layer_size = self.config[i][0]
                    
                    w = np.random.randn(current_layer_size,previous_layer_size) * np.sqrt(2. / previous_layer_size)
                    b = np.zeros([current_layer_size,1])
                    self.thetas.append(np.append(b,w, axis=1))

                    activation = Activation_Functions.factory(self.config[i][1])
                    self.activations.append(activation)

    def fit(self, X:list, y:list, X_test=None, y_test=None, epochs:int=1, alpha:float=0.01, llambda:float = 0, batch_size:int=0, shuffle:bool=True, keep_best:bool=False):
        """# Train
        Train network with back propagation using stochastic gradient descent.

        ### Args:
            - X: Training data
            - y: Training Targets
        
        ### Kwargs:
            - X_test: Test data for classification problems.  Will automatically enable accuracy checking after each epoch
            - y_test: Test targets for classification problems.  Will automatically enable accuracy checking after each epoch
            - epochs: How many itterations to process all training data
            - alpha: Learning Rate
            - batch_size: how many samples to use in each batch.  If set to 0 then use all data.
            - llambda: Regularization rate
            - shuffle: Shuffle data between epochs
            - keep_best: use thetas that had the lowest cost (J) 
        """
        X = np.array(X)
        y = np.array(y)
        m = len(y)
        min_J = None


        if not self.thetas:
            self.initialize(X)


        if  0 >= batch_size or batch_size > m:
            batch_size = m
        batches = m // batch_size
    
        for epoch in range(epochs):

            if shuffle:
                X, y = Utils.shuffle(X, y)

            for batch in range(batches):

                i_start = batch*batch_size 
                i_end = i_start + batch_size

                X_batch = X[i_start:i_end]
                y_batch = y[i_start:i_end]
                
                z, a = self.feed_forawrd(X_batch)
                grad = self.back_propagate(y_batch, z, a, llambda = llambda)
                self.update_parameters(grad, alpha)

                if keep_best:
                    if min_J == None or J < min_J:
                        min_J = J
                        _, best_thetas, _ = self.save()

            J = self.compute_loss(y_batch, a, llambda = llambda)
            if X_test is not None and y_test is not None:
                self.compute_accuracy(X, y, X_test, y_test)
            self.J_history['J'].append(J)
            self.epochs += 1

        if keep_best:
            self.thetas = best_thetas


    def predict(self, X:list):
        """# Predict
        Return hypothesis for a given input
        
        ### Args:
            - X: Input

        ### Returns:
            Prediction h(X)
        """
        X = np.array(X)

        z = np.zeros(X.shape)
        a = Utils.prepend_ones(X)

        for theta, g in zip(self.thetas, self.activations):

            z = np.dot(theta, a.T).T
            a = Utils.prepend_ones(g(z))

        return a[:,1:] #h(X)

    def feed_forawrd(self, X:list):
        """# Feed Forward
        Feed training data through network, similar to predeict, but returns pre-activations and activations for all layers for use in back propagations
        
        ### Args:
        - X: training data
        
        ### Returns:
        - Z, A: Pre-activations and Activations for each layer
        
        """
        X = np.array(X)

        z = np.zeros(X.shape)
        a = Utils.prepend_ones(X)

        Z = [z]
        A = [a]

        for theta, g in zip(self.thetas, self.activations):

            z = np.dot(theta, a.T).T
            a = Utils.prepend_ones(g(z))

            Z.append(z)
            A.append(a)

        A[-1] = a[:,1:]
        return Z, A


    def back_propagate(self, y, z, a, llambda = 0):
        """# Back Propogate
        Calculate gradient of each layer in the network
        
        ### Args:
        - y: Training targets
        - z: Layer pre-activations
        - a: Layer activations
        
        ### Kwargs:
        - llambda: Regularization amount
        
        """

        m = len(y)
        theta = self.thetas
        activations = self.activations

        d = [[] for _ in theta]
        grad = []

        indexs = list(range(len(theta)))[1:]
        indexs.reverse()
        
        d.append(a[-1] - y)
        
        # calculate error
        for i in indexs:
            g = activations[i-1]
            d[i] = np.dot(d[i+1], theta[i][:,1:]) *  g(z[i], derivative=True)
    
        # calculate gradient
        for i, a in enumerate(a[:-1]):
            gw = (1/m) * np.dot(a[:,1:].T, d[i+1]) + (llambda/m) * theta[i][:,1:].T
            gb = (1/m) * np.sum(d[i+1], axis=0, keepdims=True)
            g = np.append(gb, gw, axis=0).T


            grad.append(g) 

        return grad


    def compute_loss(self, y:list, a:list, llambda: float=0):
        """# Compute Loss
        Compute the cost, or loss of the network to determine training performance
        
        ### Args:
        - y: targets
        - a: layer activations
        - llambda: regularization amount
        
        ## Returns:
        - J: The loss of the network
        """
        
        y = np.array(y)
        m = len(y)

        reg = 0
        for theta in self.thetas:
            reg += (llambda/(2*m)) * np.sum(theta[:,1:])**2

        J = (1/m) * np.sum(-y*np.log(a[-1]) - (1-y) * np.log(1-a[-1])) + reg
        return J


    def compute_accuracy(self, X:list, y, X_test:list, y_test:list):
        """# Compute Accuracy
        Compute the accuracy of the network for classification problems
        
        ### Args:
        - X: Training Data
        - y: Training Targets
        - X_test: Test Data
        - y_test: Test Targets
        
        ### Returns:
        - None, values are stored in class variable "accuracy_history"
        
        """
        train = np.sum(np.argmax(self.predict(X), axis=1) == np.argmax(y, axis=1)) / len(y)
        test =  np.sum(np.argmax(self.predict(X_test), axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        self.accuracy_history['Train'].append(train)
        self.accuracy_history['Test'].append(test)

    def update_parameters(self, grad, alpha):
        """# Update Parameters
        Update the values of theta based off the gradient and learning rate
        
        ### Args:
        - grad:  Gradients for theta calculated through back prop
        - alpha: Learning rate
        
        
        """
        
        for theta, g in zip(self.thetas, grad):
            theta -= alpha * g


    def save(self):
        """# Save
        return network parameters
        
        ### Returns:
            weights, biases
        """

        config = [config.copy() for config in self.config]
        thetas = [theta.copy() for theta in self.thetas]

        return config, thetas, self.activations.copy(), 


    def load(self, config, thetas, activations):
        """# Load
        load network parameters
        
        ### Args:
            - weights_biases: saved parameters
        """
        self.config = config
        self.thetas = thetas
        self.activations = activations


    def copy(self):
        nn = Neural_Network()
        config, thetas, activations = self.save()
        nn.load(config, thetas, activations)
        return nn


    def mutate(self, rate=0.25, scale = 0.1):
        """#Mutate 
        Randomly select network parameters and change them by a random amount
        
        ### Kwargs:
            - rate: likelyhood is a specific parameter is changed
            - scale: magnitude of the change will be if selected
        """

        for i, theta in enumerate(self.thetas):
            m, n = theta.shape
            mutation = np.random.normal(loc=0, scale=scale, size=(m, n))
            self.thetas[i] = np.where(np.random.rand(m, n) <= rate, theta+mutation, theta)


    def crossover(self, spouse):
        """# Crossover
        Create a new neural network based on the parameters of two parents.
        Each parameter has a 50% probability to be selected from either parent.
        
        ### Args:
            - spouse: the network to crossover with

        ### Returns:
            Child neural network"""
        
        child = spouse.copy()

        for i, (t1, t2) in enumerate(zip(self.thetas, spouse.thetas)):
            m, n = t1.shape
            child.thetas[i] = np.where(np.random.rand(m, n) >= .5, t1.copy(),t2.copy())

        return child


    def __call__(self, X:list):
        return self.predict(X)

    class Population:
        def __init__(self, population_size:int, input_size:int, layer_config:list) -> None:
            '''#Create Population
            A population of networks with random initializations for use with a genetic algorithm
            
            ### Args:
                - population_size: Amount of agents to spawn
                - input_size: Amount of nodes in the input layer
                - layer_config: List definining the hidden layers and output specifications.
                    example: 2 hidden ReLu layers with 10 softmax output:
                        [[16, 'relu'],[16, 'relu'],[10, 'softmax']]
                    example: 4 hidden ReLu layers with 2 softmax output:
                        [[16, 'relu'],[32, 'relu'],[64, 'relu'],[8, 'relu'],[2, 'softmax']]
            '''
            self.agents =  [Neural_Network(input_size = input_size, config = layer_config) for _ in range(population_size)]
            self.population_size = population_size
            self.layer_config = layer_config

        def __call__(self):
            return self.agents

        def get_agent(self, index:int):
            return self.agents[index]

        def crossover(self, fitnesses: list) -> None:
            from random import choices
            '''# Crossover
            For all members of a population, weighted stochastic selection of two individuals based on fitness.
            Parameters are then mixed with a 50% probability to be from either parent.

            ### Args:
                - population: list of agent population 
                - fitness: list of fitness results for agents

            ### Retuns:
                Child newtworks
            '''
            new_population = []
            for _ in self.agents:
                network_x, network_y = choices(self.agents, weights=fitnesses, k=2)
                new_population.append(network_x.crossover(network_y))
            self.agents  = new_population

        def mutate(self, rate: float = 0.01, scale: float = 0.1) -> None:
            '''#Mutate
            In place stochastic adjustment of weights and biases for all members of a population.
            
            ### Args:
                - population: list of agent populations

            ### Kwargs:
                - rate: odds that any given weight or bias is updated: 0.01 being 1% change to mutate, 0.1 being 10%
                - scale: magnitude of the mutation
            '''
            for network in self.agents:
                network.mutate(rate=rate, scale=scale)


class Normal_Equation:
    """# Normal Equation
    Alternative method to fit theta to minimize J.  Will always find best values of theta, but is slow for large data sets.
    Good for realativly small datasets.  Linear Regression may be faster.    
    """
    
    def fit(self, X,y):
        """# Fit
        Fit theta useing the normal equation.  Good for realativly small datasets.
        Very slow for large data.
        ### Args:
        - X: Training Date
        - y: Training Targets

        ## Returns: Instance of Normal class
        """
        X = np.array(X)
        y = np.array(y)

        X = Utils.prepend_ones(X)
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

    def predict(self, X):
        """# Predict

        ### Args:
        - X: Data to predict

        ## Returns:
        - Predicted results
        """
        X = np.array(X)
        X = Utils.prepend_ones(X)
        return X.dot(self.theta)

class SGD:
    """# Stochastic Gradient Descent
    Implemented with a scikit-learn like api for ease of use."""

    def __init__(self, function = 'logistic') -> None:
        self.function = Activation_Functions.factory(function)
        self.feature_scaling = False

    def compute_cost(self, X:list, y:list) -> float:
        """# Compute Cost
        Calculate the cost (error) of the predictions against the targets given the current values of theta.

        ### Args:
            - X: [np.array] Training Data
            - y: [np.array] Training Targets
        
        ### Return:
            - [float] Cost (J)
        """
        m = len(y)
        h = self.function(np.dot(X,self.theta))
        J = (1/m) * np.nansum(-y*np.log(h) - (1-y)*np.log(1-h))  +  (self.llambda/(2*m) * np.nansum(np.power(self.theta[1:],2)))
        return J
         
    def predict(self, X:list) -> list:
        """# Predict
        Predict a new hyphothis for an input.  If scaling was enabled durring training, X will be automatically scaled before prediction.
        
        ### Args:
            - X: [np.array] Data 

        ### Returns:
            - y: [np.array] Hypothesis """

        if self.feature_scaling:
            X = self.scaler.transform(X)
        X = Utils.prepend_ones(X)
        z = np.dot(X, self.theta)
        return self.function(z)

    def fit(self, X:list, y:list,
            alpha:float = .01, n_epochs:int = 100, batch_size:int = None, llambda:float = 0.,
            theta:list = None, warm_start:bool = False, keep_best:bool = False, feature_scaling:bool = False, shuffle:bool= False):
        """# Fit
        Train the model with a dataset using a stochastic gradient descent optimizer.
        
        ### Args:
            - X: [np.array] Training Data 
            - y: [np.array] Training Target

        ### Kwargs:
            - alpha: [float default: 0.1] Learning Rate 
            - n_epochs: [int default: 100] How many times to run through the entire training set 
            - batch_size: [int default: None] How many training examples to process in a batch. If None then run in BGD mode. 
            - llambda: [float default: 0] regularization amount.  Increasing llambda strengthens the regularization effect
            - theta: [np.array default: None] model coefficient's if loading a pre-trained model 
            - warm_start: [bool default: False] Keep parameters and continue training 
            - feature_scaling: [bool default: False] Learn mu (mean) and sigma (standard deviation) of training data.  Scale X using mu and theta. Scaling will be applied to X in predict method.
            - shuffle: [bool default: False] Shuffle data after each epoch 

        ### Raises:
            - ValueError: if shape of theta does not conform with existing weights

        ### Returns:
            - Instance of class SGD
        """

        y = y*1 #convert boolean to int
        self.mu = 0
        self.sigma = 1
        self.min_J = None
        self.llambda = llambda 

        if feature_scaling:
            self.feature_scaling = True
            self.scaler = Utils.Scaler()
            X = self.scaler.fit_transform(X)

        X = Utils.prepend_ones(X)

        if not warm_start:
            self.theta = np.zeros(X.shape[1])
            self.J_history = np.array([])

        if theta:
            if np.shape(self.theta) != np.shape(self.theta):
                raise ValueError('Shape of theta does not conform with existing weights')
            self.theta = theta

        m = len(y)
        if not batch_size or batch_size > m:
            batch_size = m
        n_batchs = m // batch_size
        J_history = np.zeros(n_epochs * n_batchs)
        
        for epoch in range(n_epochs):
            if shuffle:
                (X, y) = Utils.shuffle(X,y)
            for batch in range(n_batchs):

                indx_batch_start = (batch)*batch_size
                indx_batch_end = indx_batch_start + batch_size

                X_batch = X[indx_batch_start:indx_batch_end]
                y_batch = y[indx_batch_start:indx_batch_end]
   
                z = np.dot(X_batch, self.theta)

                theta_temp = self.theta
                theta_temp[0] = 0

                grad = ((1/m) * np.dot(self.function(z)-y_batch,  X_batch) ) + llambda*theta_temp

                self.theta -= alpha * grad

                J = self.compute_cost(X_batch, y_batch)
                if keep_best:
                    if self.min_J == None or  J < self.min_J:
                        self.min_J = J
                        self.best_theta = np.copy(self.theta)

                J_history[(batch)+(epoch*n_batchs)] = J

        self.J_history = np.append(self.J_history, J_history)
        if keep_best:
            self.theta = np.copy(self.best_theta)

        return self

class Logistic_Regression():
    """# Logistic Regression
    Classifier using stochastic gradient descent"""
    
    def fit(self, X:list, y:list,
                alpha:float = .01, n_epochs:int = 100, batch_size:int = 300, llambda:float = 0.,
                theta:list = None, warm_start:bool = False, keep_best:bool = False, feature_scaling:bool = False, shuffle:bool= False):
            """# Fit
            Train the model with a dataset using a stochastic gradient descent optimizer.
            
            ### Args:
                - X: [np.array] Training Data 
                - y: [np.array] Training Target

            ### Kwargs:
                - alpha: [float default: 0.1] Learning Rate 
                - n_epochs: [int default: 100] How many times to run through the entire training set 
                - batch_size: [int default: 300] How many training examples to process in a batch 
                - llambda: [float default: 0] regularization amount.  Increasing llambda strengthens the regularization effect
                - theta: [np.array default: None] model coefficient's if loading a pre-trained model 
                - warm_start: [bool default: False] Keep parameters and continue training 
                - feature_scaling: [bool default: False] Learn mu (mean) and sigma (standard deviation) of training data.  Scale X using mu and theta. Scaling will be applied to X in predict method.
                - shuffle: [bool default: False] Shuffle data after each epoch 

            ### Raises:
                - ValueError: if shape of theta does not conform with existing weights

            ### Returns:
                - Instance of class Logistic_Regression
            """
            if not warm_start:
                self.clfs = []
                self.threshold_multipliers = []
                self.ohe = Utils.One_Hot_Encoder()
                self.ohe.fit(y)
                
            y = self.ohe.transform(y)
            
            n_keys = len(self.ohe.keys)
            n_clfs = 1
            if n_keys >= 1:
                n_clfs = n_keys                

            for i in range(n_clfs):
                    if warm_start:
                        clf = self.clfs[i]
                    else:
                        clf = SGD(function = 'logistic')
                    clf.fit(X, y[:,i],
                            alpha = alpha, n_epochs = n_epochs, batch_size = batch_size, llambda = llambda,
                            theta = theta, warm_start = warm_start, feature_scaling = feature_scaling, keep_best=keep_best, shuffle = shuffle)
                    if not warm_start:
                        self.clfs.append(clf)
                        self.threshold_multipliers.append(1)
            return self
    
    def predict(self, X, return_label=True):
        """# Predict
        Predict a new hyphothis for an input.  If scaling was enabled durring training, X will be automatically scaled before prediction.
        
        ### Args:
            - X: [np.array] Data 
        
        ### Kwargs:
            - return_label: [Bool default:True] When true, hypothesis is returned as class labels.
            When false data is returned as vectors of probabilities for each class.

        ### Returns:
            - y: [np.array] Hypothesis
            """
        if len(self.clfs) == 1:
            preds = np.zeros((len(X),2))
            preds[:,1] = 0.5
        else:
            preds = np.zeros((len(X),len(self.clfs)))

        for indx, clf  in enumerate(self.clfs):
            preds[:,indx] = clf.predict(X)
            
        preds = preds * self.threshold_multipliers
        
        if return_label:
            preds = np.array(np.max(preds, axis=1) == preds.T, dtype=int).T
            return self.ohe.reverse_transform(preds)

        return preds


class Linear_Regression(SGD):
    """# Linear Regression
    Regression using stochastic gradient descent"""
    def __init__(self, function='linear') -> None:
        super().__init__(function=function)


class PCA():
    """# Principal Component Analysis
    Used for dementionality reduction    
    """

    def __init__(self) -> None:
        self.feature_scaling = False

    def fit(self, X:list, features:float, feature_scaling:bool = False):
        """# Fit
        Fit model to data

        ### Args:
        - X: Input data to be reduced
        - features: If greater or equal to 1 set the amount of features to reduce to.  If less than 1, treated as the amount of variance to maintain,
        the model will calculate the minimum amount of features using the cumulative total variance explained of the eigenvalues

        ### Kwargs:
        - feature_scaling: [Bool default False] if true, the model will automatically train a scaler, and scale all inputs.

        ## Returns:
        - Instance of the PCA class        
        """
        X = np.array(X)

        if feature_scaling:
            self.feature_scaling = True
            self.scaler = Utils.Scaler()
            X = self.scaler.fit_transform(X)

        m, _ = X.shape
        
        self.sigma = (1/m) * np.dot(X.T, X) # Calculate covariance matrix
        w, u = np.linalg.eig(self.sigma) # Compute eigenvectors and eigenvalues
        self.u = np.real(u) # eigenvectors
        self.w = np.real(w) # eigenvalues

        self.total_variance_explained = (w / np.sum(w))

        # Set number of features to reduce to
        if features >= 1: # user selects k
            self.k = int(features)
        else: #user selects minimum cumulative variance
            for i, cum_variance in enumerate(np.cumsum(self.total_variance_explained)):
                if features < cum_variance:
                    self.k = i+1
                    break
        return self

    def transform(self, X) -> list:
        """# Transform
        Transformed data using fitted model

        ### Args:
        - X: Input data to be reduced

        ## Returns:
        - Transformed data    
        """         

        if self.feature_scaling:
            X = self.scaler.transform(X)
        u_reduce = self.u[:,:self.k].T
        z = np.dot(u_reduce,X.T).T
        return z

    def fit_transform(self, X:list, features:float, feature_scaling:bool = False) -> list:
        """# Fit Transform
        Fit model to data to return transformed data

        ### Args:
        - X: Input data to be reduced
        - features: If greater or equal to 1 set the amount of features to reduce to.  If less than 1, treated as the amount of variance to maintain,
        the model will calculate the minimum amount of features using the cumulative total variance explained of the eigenvalues

        ### Kwargs:
        - feature_scaling: [Bool default False] if true, the model will automatically train a scaler, and scale all inputs.

        ## Returns:
        - Transformed data    
        """           

        self.fit(X, features, feature_scaling = feature_scaling)
        return self.transform(X)

    def reverse_transform(self, z) -> list:
        """# Reverse Transform
        Recover data back to its original dementions.  PCA is lossy so details will be reduced.

        ### Args:
        - z: Transformed data

        ## Returns:
        - Recovered data  
        """    
        X_rec = np.dot(self.u[:,:self.k],z.T).T
        return X_rec

class K_Means:
    """# K Means
    centroid clustering    
    """

    def predict(self, X):
        """# Predict
        Returns a class assignment for records

        ### Args:
        - X: Data to classify

        ### Returns:
         - Class predictions
        """
        K = len(self.centroids)
        m, _ = X.shape 
        euc_d = np.zeros((m,K))

        for k, _ in enumerate(self.centroids):
            euc_d[:,k] = np.sqrt(np.sum((X - self.centroids[k])**2, axis = 1))

        idx = np.argmin(euc_d, axis=1)
        return idx

    
    def fit(self, X, K, epochs=1, batch_size = None, shuffle = False):
        """# Fit
        Fit model to data

        ### Args:
        - X: Data to cluster
        - K: Number of centroids to cluster

        ### Kwargs:
        - epochs: Amount of times to itterate over full data set
        - batch_size: Amount of records to train in each batch
        - shuffle: shuffle data between ephochs

        ### Returns:
        - Instance of the K_Means class        
        """

        m, n = X.shape
        max = np.max(X)
        self.centroids = np.random.rand(K,n)*max

        for _ in range(epochs):
            if shuffle:
                p = np.random.permutation(m)
                X = X[p]

            if not batch_size:
                batch_size = m

            n_batches = m // batch_size
            X = X[:n_batches * m]

            for b in range(n_batches):
                batch_start = b * batch_size
                batch_end = (b+1) * batch_size
                batch = X[batch_start:batch_end]

                preds = self.predict(batch)
                for i, _ in enumerate(self.centroids):
                    mask = (np.ones((n,batch_size))*(preds != i)).T

                    self.centroids[i] = np.asarray(np.ma.masked_array(batch,mask=mask).mean(axis=0))
        return self

class Utils:

    def confusion(y:list, preds:list) -> list:
        """# Confusion
        Return the confusion matrix for the predictions of a classifier.  Also returns key diagnostic information
        
        ### Args:
            - y:
            - preds:
            
        ### Returns
            - Dictonary of values
            {'accuray',
            'macro avg': {precision, recall, f1},
            'weighted avg': {precision, recall, f1},
            'class': {labels, precision, recall, f1},
            'confusion matrix'}
        """

        le = Utils.Label_Encoder()
        le.fit(y)
        n_classes = len(le.labels)
        n_samples = len(y)

        targets = np.array(le.transform(y))
        preds = np.array(le.transform(preds))
        
        matrix = np.zeros((n_classes,n_classes))
        for p,t in zip(preds, targets):
            matrix[p,t] += 1

        tp = sum(np.eye(n_classes) * matrix)
        fn = sum((np.eye(n_classes)==0) * matrix)
        fp = np.sum((np.eye(n_classes)==0) * matrix, axis = 1)

                
        ohe = Utils.One_Hot_Encoder()
        weights = np.sum(ohe.fit_transform(y), axis = 0)


        precision = np.nan_to_num(tp / (tp + fp))
        recall = np.nan_to_num(tp / (tp + fn))
        f1 = np.nan_to_num((2 * precision * recall) / (precision + recall))


        return {

            'accuracy': sum(preds == np.array(targets)) / n_samples,
            
            'macro avg':{
                'precision': np.mean(precision),
                'recall': np.mean(recall),
                'f1': np.mean(f1)
            },

            'weighted avg':{
                'precision':np.sum(precision*weights) / n_samples,
                'recall': np.sum(recall*weights) / n_samples,
                'f1': np.sum(f1*weights) / n_samples
            },


            'class':{
                'labels': le.labels,
                'precision':precision,
                'recall':recall,
                'f1':f1
            },

            'confusion matrix':matrix
        }

    def under_sample(X:list, y:list) -> list:
        """# Under Sample
        Find the class that has the fewest samples then decrease 
        the number of samples in all other classes such that all classes
        have the same amount of samples.

        ### Args:
            - X: Data
            - y: Targets
        
        ### Returns:
            - Under sampled X, y
        """
        ohe = Utils.One_Hot_Encoder()
        oh = ohe.fit_transform(y)
        samples = np.sum(oh, axis=0)

        indexs = []
        to_remove = samples - min(samples)

        while sum(to_remove) > 0:
            for indx, item in enumerate(oh):
                if np.all(to_remove - item >= 0):
                    
                    indexs.append(indx)
                    to_remove -= item
                    
        indexs.reverse()
        y = np.delete(y,indexs, axis=0)
        X = np.delete(X,indexs, axis=0)
        
        return X, y


    def prepend_ones(X:list) -> list:
        """# Prepend Ones
        Prepends each row of data with ones to act as static coefficient to theta[0].

        ### Args:
            - X: [np.array] Data

        ### Returns:
            - [np.array] Data
        """
        X_ones = np.ones((np.shape(X)[0], np.shape(X)[1]+1))
        X_ones[:,1:] = X
        return X_ones

    def prepend_zeros(X:list) -> list:
        """# Prepend Ones
        Prepends each row of data with zeros.

        ### Args:
            - X: [np.array] Data

        ### Returns:
            - [np.array] Data
        """
        X_ones = np.zeros((np.shape(X)[0], np.shape(X)[1]+1))
        X_ones[:,1:] = X
        return X_ones

    def shuffle(X:list, y:list) -> list:
        """# Shuffle
        Reorder two arrays of the same length in unison and return the randomized results.

        ### Args:
            - X: [np.array] Training data
            - y: [np.array] Targets

        ### Returns:
            - X: [np.array] Shuffled training data 
            - y: [np.array] Shuffled Targets
        """
        m = len(y)
        assert len(X) == m
        p = np.random.permutation(m)
        return (X[p], y[p])

    def train_test_split(X:list, y:list, train_size:int, shuffle:bool = False) -> list:
        """# Train Test Split
        Split training data into training and test sets.

        ### Args:
            - X: [np.array] Data
            - y: [np.array] Targets
            - train_size: [int] Number of records to be in train set.  Remainder is added to test set

        ### Kwargs:
            - shuffle: shuffle data before splitting

        ### Returns:
            - (X_train, y_train), (X_test, y_test)
        """
        if shuffle:
            X, y = Utils.shuffle(X, y)

        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        return (X_train, y_train), (X_test, y_test)

    class Scaler:
        """# Scaler
        Standardize input data to avoid shallow gradiant issues with optimizers"""


        def fit(self, X:list) -> list:
            """# Fit
            Learn the mean and standard deviation of the training data and store in mu and sigma.

            ### Args:
                - X: [np.array] Data
            
            ### Returns:
                - Instance of class Scaler
            """

            self.mu = np.mean(X)
            self.sigma = np.std(X)
            return self

        def transform(self, X:list) -> list:
            """# Transform
            Perform standardization by centering and scaling.

            ### Args:
                - X: [np.array] Data

            ### Returns:
                - [np.array] Scaled Data
            """

            X = np.divide(X - self.mu,  self.sigma)
            return X

        def reverse_transform(self, X:list) -> list:

            return X * self.sigma + self.mu

        def fit_transform(self, X:list) -> list:
            """# Fit Transform
            Fit to data, then returned scaled data

            ### Args:
                - X: [np.array] Data

            ### Returns:
                - [np.array] Scaled Data        
            """

            self.fit(X)
            return self.transform(X)


    class Label_Encoder:
        """# Label Encoder
        Create a mapping for unique values in target data to an integer index."""

        def fit(self, y:list):
            """# Fit
            Extract labels from data and store in class variable "keys"
            Extract values from data and store in class variable "values"

            ### Args:
                - y: [np.array] Targets

            ### Returns:
                - Instance of Label_Encoder class     
            """

            self.labels = np.unique(y)
            self.keys = {y_val:i for i, y_val in enumerate(self.labels)}
            self.values = {i:y_val for i, y_val in enumerate (self.labels)}
            return self

        def transform(self, y:list) -> list:
            """# Transform
            Change labeled data to integer values based on mappings

            ### Args:
                - y: [np.array] Targets

            ### Returns:
                - [np.array] Mapped Targets     
            """
            return np.array([self.keys[key] for key in y])

        def fit_transform(self, y:list) -> list:
            """# Fit Transform
            Create mapping of labels to integer values, then returned mapped data.

            ### Args:
                - y: [np.array] Targets

            ### Returns:
                - [np.array] Mapped Targets       
            """

            self.fit(y)
            return np.array(self.transform(y))

        def reverse_transform(self, y:list) -> list:
            """# Reverse Transform
            Return list back to the labeled format

            ### Args:
                - y: [np.array] Mapped Targets

            ### Returns:
                - [np.array] Targets       
            """

            return [self.values[value] for value in y]


        def __call__(self, y):
            return self.transform(y)

    class One_Hot_Encoder:

        def fit(self, y:list):
            """# Fit
            Extract list of unique values to keys

            ### Args:
                - y: [np.array] Targets

            ### Returns:
                - Instance of One_Hot_Encoder class     
            """
            self.keys = np.unique(y)
            return self

        def transform(self, y:list) -> list:
            """# Transform
            Change labeled data to matrix where each column represents a value and each
            each row has one value of 1 in the column representing the original value.

            ### Args:
                - y: [np.array] Targets

            ### Returns:
                - [np.array] Vectorized Targets     
            """
            return np.array(np.array(y)[...,None] == self.keys[None, ...], dtype=int)

        def fit_transform(self, y:list) -> list:
            """# Fit Transform
            Extract list of unique values to keys, then returned vectorized data.

            ### Args:
                - y: [np.array] Targets

            ### Returns:
                - [np.array] Vectorized Targets       
            """
            self.fit(y)
            return self.transform(y)

        def reverse_transform(self, y:list) -> list:
            """# Reverse Transform
            Return list back to the labeled format

            ### Args:
                - y: [np.array] Vectorized Targets

            ### Returns:
                - [np.array] Targets       
            """
            y = np.array(y)

            y = np.equal(np.max(y, axis=1, keepdims=True), y)*1 #convert probabilities into one hot
            n_keys = len(self.keys)
            identity = np.array(range(n_keys))
            index = np.sum(np.multiply(y,  identity), axis=1)
     
            return self.keys[index]
