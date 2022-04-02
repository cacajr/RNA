from perceptron.perceptron import Perceptron
import numpy as np
import pandas as pd


class MLP:
    def __init__(self, config_layers = [2, 2], bies = -1, eta = 0.001, epoch = 100, W = [], activation_function = 'sigmoid'):
        self.__perceptrons = [
            [
                Perceptron(bies, eta, 1, W, activation_function) 
                for _ in range(num_perceptrons)
            ]
            for num_perceptrons in config_layers
        ]
        self.__out = Perceptron(bies, eta, 1, W, activation_function)
        self.eta = eta
        self.epoch = epoch

    
    def get_perceptrons(self):
        return self.__perceptrons


    def get_out(self):
        return self.__out


    def fit(self, X, y):
        for _ in range(self.epoch):
            for x, yn in zip(X.values, y.values):
                inputs = pd.DataFrame([x], columns = X.columns)
                y_inputs = pd.Series(yn)
                for layer in self.__perceptrons:
                    predicts = []
                    for perceptron in layer:
                        perceptron.fit(inputs, y_inputs, False) # treino sem aplicar a regra de aprendizado
                        predicts.append(perceptron.predict(x))
                    inputs = pd.DataFrame([predicts], columns = X.columns)

                # MLP Out
                predicts = []
                for perceptron in self.__perceptrons[-1]:
                    predicts.append(perceptron.predict(x))
                inputs = pd.DataFrame([predicts], columns = X.columns)
                self.__out.fit(inputs, y_inputs, False)
                err = yn - self.__out.predict(x)

                # Learn Rule (Miscalculation)
                u = self.__out.predict(x, False)
                delta = self.eta * err * self.__der_sigmoid(u)
                new_W = self.__out.get_weight()
                new_W[1:] += delta * x
                new_W[0] -= delta
                self.__out.set_weight(new_W)

                # Backpropagation
                for layer in self.__perceptrons:
                    for perceptron in layer:
                        u = perceptron.predict(x, False)
                        yn = perceptron.predict(x)
                        sum_err = 0
                        for wn in perceptron.get_weight():
                            sum_err += err * self.__der_sigmoid(yn) * wn
                        delta = self.eta * sum_err * self.__der_sigmoid(u)
                        new_W = perceptron.get_weight()
                        new_W[1:] += delta * x
                        new_W[0] -= delta
                        perceptron.set_weight(new_W)
                
    
    def __sigmoid(self, u):
        return 1/(1 + np.exp(-u))


    def __der_sigmoid(self, u):
        return self.__sigmoid(u)*(1 - self.__sigmoid(u))

    
    def predict(self, x):
        return self.__out.predict(x)


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
