
class Inference():

    def __init__(self):
        pass

    def makeInference(self):
        raise NotImplementedError
    
    def addEvidence(self):
        raise NotImplementedError
    
    def removeEvidence(self):
        raise NotImplementedError
    
    def currentEvidence(self):
        raise NotImplementedError
    
    def posterior(self):
        raise NotImplementedError
    