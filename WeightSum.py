class WeightedSum:
    def __init__(self):
        self.parent = None
        self.children = []
        self.sum = 0

    def addParent(self, parent):
        self.parents.append(parent)
    
    def addChild(self, child): 
        self.children.append(child)

    def recordSum(self, sum):
        self.sum += sum

    def feedForward(self):
        self.parent.recordValue(self.sum)

    # TODO : implement the feedBackward function
    # This function is used to calculate the gradient of the weighted sum function
    def feedBackward(self):
        pass