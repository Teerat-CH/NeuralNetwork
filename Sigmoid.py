import math

class Sigmoid:
    def __init__(self):
        self.parents = [] # list of weighted sums node
        self.weights = [] # list of weights for each parent
        self.children = None
        self.value = None

    def addParent(self, parent, weight):
        self.parents.append(parent)
        self.weights.append(weight)

    def addChild(self, child):
        self.children.append(child)

    def recordValue(self, value):
        self.value = 1 / (1 + math.exp(-self.value))
    
    def feedForward(self):
        for i in range(len(self.parents)):
            self.parents[i].recordSum(self.weights[i] * self.value)

    # TODO : implement the feedBackward function
    # This function is used to calculate the gradient of the sigmoid function
    def feedBackward(self):
        pass