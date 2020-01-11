import abc

class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, batch, asyncCall = None):
        """
        @param batch: a batch of encoded game states, encoding happens by the GameState implenentation, typically this should be tensors of some kind
        @param asyncCall: if set this is called in a moment where the cpu would otherwise by idle, waiting for gpu work. 
            If the Policy does no gpu work this shall just be called at the end of this method.
        @return: list of tuples (move distribution, win probabilities)
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
    def fit(self, data, epochs):
        """
        Fits the policy to the data given. Changes the policy UUID. Does not require the policy
        to randomize parameters before fitting, so e.g. a network can learn based on previous parameters.
        @param data: a list of dict() objects with data as defined by the GameReporter interface
        @param epochs: How many epochs to train
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