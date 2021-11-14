import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class Activation_Functions:
    """# Activation Function Library"""

    def sigmoid(z:list) -> list:
        return 1/(1+np.exp(-z))

    def linear(z:list) -> list:
        return z

    def factory(function:str):
        """# Activation Factory
        Factory that returns an activation function.
        
        ### Suported activation functions:
            - sigmoid: used for logistic regression
            - linear: used for linear regression
        
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
        else:
            raise ValueError('"{}" is not a supported function.'.format(function))

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
            n_keys = len(self.keys)
            identity = np.array(range(n_keys))
            index = np.sum(np.multiply(y,  identity), axis=1)
     
            return self.keys[index]