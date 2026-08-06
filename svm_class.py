# svm_class.py

import numpy as numpy

class LinearSVM:

    def __init__(self, learningRate = 1e-3, lambdaParam = 1e-2, iteration = 1000):

        self.eta = learningRate
        self.lambdaParam = lambdaParam
        self.iteration = iteration
        self.w = None
        self.b = None
        

    def fit(self, x, y):

        sample, feature = x.shape
        y = numpy.where(y <= 0, -1, 1) # convert label to {-1, +1}

        self.w = numpy.zeros(feature)
        self.b = 0.0

        for epoch in range(self.iteration):

            for idx, xi in enumerate(x):

                yi = y[idx]
                margin = yi * (numpy.dot(self.w, xi) + self.b)

                if margin >= 1:

                    wGrad = self.lambdaParam * self.w
                    bGrad = 0

                else:

                    wGrad = self.lambdaParam * self.w - yi * xi
                    bGrad = -yi

                
                # update

                self.w -= self.eta * wGrad
                self.b -= self.eta * bGrad

        return self


    def decisionFunction(self, x):

        return numpy.dot(x, self.w) + self.b

    
    def predict(self, x):

        approx = self.decisionFunction(x)
        yHat = numpy.sign(approx)

        return numpy.where(yHat == -1, 0, 1)



class MultiClassSVM:

    def __init__(self, learningRate = 1e-3, lambdaParam = 1e-2, iteration = 3000):

        self.learningRate = learningRate
        self.lambdaParam = lambdaParam
        self.iteration = iteration
        self.model = {}
        self.classes = None

    def fit(self, x, y):

        self.classes = numpy.unique(y)

        for cls in self.classes:

            yBin = numpy.where(y == cls, 1 , 0) # this class = 1, rest = 0
            model = LinearSVM(self.learningRate, self.lambdaParam, self.iteration)
            model.fit(x, yBin)
            self.model[cls] = model

        return self

    def decisionScore(self, x):

        # return dict {class: f(x) score} for point (or batch of points)
        return {cls: m.decisionFunction(x) for cls, m in self.model.items()}

    def predict(self, x):

        score = numpy.column_stack([self.model[cls].decisionFunction(x) for cls in self.classes])
        bestidx = numpy.argmax(score, axis = 1)

        return self.classes[bestidx]