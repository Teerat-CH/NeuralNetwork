import Perceptron

class Layer:
    def __init__(self, size):
        self.perceptrons = []
        for i in range(size):
            self.perceptrons.append(Perceptron())

    def connectTo(layer):
        for firstLayerPerceptron in self.perceptrons:
            for secondLayerPerceptron in layer.perceptrons:
                firstLayerPerceptron.addParent(secondLayerPerceptron)
                secondLayerPerceptron.addChild(firstLayerPerceptron)