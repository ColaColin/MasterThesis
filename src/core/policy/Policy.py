import abc

class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, batch, asyncCall = None):
        """
        @param batch: a batch of encoded game states, encoding happens by the GameState implenentation, typically this should be tensors of some kind
        @param asyncCall: if set this is called in a moment where the cpu would otherwise by idle, waiting for gpu work. 
            If the Policy does not gpu work this is just called at the end of this method.
        @return: list of tuples (move distribution, win probabilities)
        """

    @abc.abstractmethod
    def getUUID(self):
        """
        fitting the policy should give it a UUID which can then be used to identify this policy
        """

    @abc.abstractmethod
    def fit(self, data):
        """
        Fits the policy to the data given.
        @param data: a list of GameState objects that will be encoded using their encode function
        @return: Nothing, the Policy is modified instead.
        """

    @abc.abstractmethod
    def load(self, packed):
        """
        load a stored policy from the given parameters
        @packed policy parameters, produced by store
        """
    
    @abc.abstractmethod
    def store(self):
        """
        @return Policy configuration, as a byte array (may be compressed) to be loaded by load()
        """