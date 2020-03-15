import abc

class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, batch, asyncCall = None):
        """
        @param batch: a batch of objects that implement GameState
        @param asyncCall: if set this is called in a moment where the cpu would otherwise by idle, waiting for gpu work. 
            If the Policy does no gpu work this shall just be called at the end of this method.
        @return: list of tuples (move distribution, win probabilities). Both are absolute: index 0 in the move distribution is the move with index 0. Index 0 in the win distribution is a draw, index 1 is player number 1, index 2 is player number 2. Again: Win probabilities are not relative to the current turn!!!
        """

    @abc.abstractmethod
    def getUUID(self):
        """
        fitting the policy should give it a UUID which can then be used to identify this policy
        """

    @abc.abstractmethod
    def reset(self):
        """
        Reset policy parameters. Changes UUID
        """

    @abc.abstractmethod
    def fit(self, data, iteration = None, iterationProgress = None, forceLr = None):
        """
        Fits the policy to the data given. Changes the policy UUID. Does not require the policy
        to randomize parameters before fitting, so e.g. a network can learn based on previous parameters,
        in fact this is not just used to learn on whole epochs of data, but is also called with single minibatches 
        in some scenarios.
        @param data: a list of dict() objects with data as defined by the GameReporter interface
        @param iteration: If given the network iteration this fit is called for. Can be None. Meant to vary e.g. the learning rate.
        @param iterationProgress: How far the current iteration has progressed. Can be None. Meant to vary e.g. the learning rate.
        @param forceLr: If set will force the learning rate to be that value.
        @return: Nothing, the Policy is modified instead.
        """

    @abc.abstractmethod
    def load(self, packed):
        """
        load a stored policy from the given parameters. Modifies this policy.
        @param packed: policy parameters, produced by store
        """
    
    @abc.abstractmethod
    def store(self):
        """
        @return Policy configuration, as a numpy byte (uint8) array to be loaded by load()
        """