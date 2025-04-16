from WeightSum import WeightSum
from Sigmoid import Sigmoid

class Perceptron:
    def __init__(self):
        self.weightedSum = WeightSum()
        self.sigmoid = Sigmoid()

        self.weightedSum.addParent(self.sigmoid)
        self.sigmoid.addChild(self.weightedSum)

    def addChild(self, child):
        self.weightedSum.addChild(child)

    def addParent(self, parent, weight):
        self.sigmoid.addParent(parent, weight)
    
    def feedForward(self):
        self.weightedSum.feedForward()
        self.sigmoid.feedForward()

    def feedBackward(self):
        self.sigmoid.feedBackward()
        self.weightedSum.feedBackward()