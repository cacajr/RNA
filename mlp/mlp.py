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
                inputs_each_layer = [inputs] # guarda a entrada de cada camada
                y_inputs = pd.Series(yn)
                for layer in self.__perceptrons:
                    predicts = []
                    for perceptron in layer:
                        perceptron.fit(inputs, y_inputs, False) # treino sem aplicar a regra de aprendizado
                        predicts.append(perceptron.predict(x))
                    inputs = pd.DataFrame([predicts], columns = X.columns)
                    inputs_each_layer.append(inputs)

                # MLP Out
                inputs = inputs_each_layer[-1] # saida da ultima camada
                self.__out.fit(inputs, y_inputs, False)
                u = self.__out.predict(x, False)
                pred = self.__out.predict(x)
                err = yn - pred

                # Learn Rule (Miscalculation)
                delta = self.eta * err * self.__der_sigmoid(u)
                
                new_W = self.__out.get_weight()
                new_W[1:] += delta * inputs.values[0]
                new_W[0] -= delta
                self.__out.set_weight(new_W)

                u = self.__out.predict(x, False)
                pred = self.__out.predict(x)
                err = yn - pred

                # Backpropagation
                for layer, inputs in zip(self.__perceptrons[::-1], inputs_each_layer[:-1][::-1]):
                    for perceptron in layer:
                        sum_err = 0
                        for wn in perceptron.get_weight():
                            sum_err += err * self.__der_sigmoid(u) * wn
                        u_ = perceptron.predict(x, False)
                        delta = self.eta * sum_err * self.__der_sigmoid(u_)

                        new_W = perceptron.get_weight()
                        new_W[1:] += delta * inputs.values[0]
                        new_W[0] -= delta
                        perceptron.set_weight(new_W)
                
    
    def __sigmoid(self, u):
        return 1/(1 + np.exp(-u))


    def __der_sigmoid(self, u):
        return self.__sigmoid(u)*(1 - self.__sigmoid(u))

    
    def predict(self, x):
        inputs = x
        for layer in self.__perceptrons:
            predicts = []
            for perceptron in layer:
                predicts.append(perceptron.predict(inputs))
            inputs = predicts

        return self.__out.predict(inputs)


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = 0 if self.predict(x) < 0 else 1 # aplicando hardlim
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
