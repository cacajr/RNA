from perceptron.perceptron import Perceptron
import numpy as np
import pandas as pd


class MLP:
    def __init__(self, config_layers = [2, 2], bies = 1, eta = 0.00001, epoch = 100, W = [], activation_function_hidden_layer = 'sigmoid', activation_function_out = 'hardlim'):
        self.__perceptrons = [
            [
                Perceptron(bies, eta, 1, W, activation_function_hidden_layer) 
                for _ in range(num_perceptrons)
            ]
            for num_perceptrons in config_layers
        ]
        self.__out = Perceptron(bies, eta, 1, W, activation_function_out)
        self.eta = eta
        self.epoch = epoch

    
    def get_perceptrons(self):
        return self.__perceptrons


    def get_out(self):
        return self.__out


    def fit(self, X, y):
        for _ in range(self.epoch):
            # shuffle
            Xy = pd.concat([X, y], axis = 1)
            Xy_shuffle = Xy.sample(frac=1)

            X = Xy_shuffle.drop(Xy_shuffle.columns[-1], axis=1)
            y = Xy_shuffle[Xy_shuffle.columns[-1]]

            for x, yn in zip(X.values, y.values):
                inputs = x
                inputs_each_layer = [inputs] # guarda a entrada de cada camada oculta
                for layer in self.__perceptrons:
                    predicts = []
                    for perceptron in layer:
                        pred = perceptron.fit(inputs, yn) # treino perceptron MLP
                        predicts.append(pred)
                    inputs = predicts
                    inputs_each_layer.append(np.array(inputs))

                # MLP Out
                inputs = inputs_each_layer[-1] # saida da ultima camada oculta
                u = self.__out.fit(inputs, yn, False)
                
                # Learn Rule (Miscalculation)
                delta_out = self.__out.get_error() * self.__der_sigmoid(u)

                new_W = self.__out.get_weight()
                w = new_W[1:] + self.eta * inputs * delta_out
                b = [new_W[0] + self.eta * self.__out.get_bies() * delta_out]
                new_W = np.concatenate([b, w])

                self.__out.set_weight(new_W)

                # ref: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
                # Backpropagation
                delta_aux = delta_out
                for layer, inputs in zip(self.__perceptrons[::-1], inputs_each_layer[:-1][::-1]):
                    # somatório dos erros nessa camada
                    sum_error = 0.0
                    for perceptron in layer:
                        sum_error += perceptron.get_weight() * delta_aux
                    sum_error = sum(sum_error)

                    # será que é assim que se propaga o delta para as demais camadas?
                    delta_aux = sum_error

                    # atualizando os erros e os pesos dessa camada
                    for perceptron in layer:
                        u_ = perceptron.predict(inputs, False)
                        delta = self.__der_sigmoid(u_) * sum_error
                        perceptron.set_error(delta)

                        new_W = perceptron.get_weight()
                        w = new_W[1:] + self.eta * inputs * delta
                        b = [new_W[0] + self.eta * perceptron.get_bies() * delta]
                        new_W = np.concatenate([b, w])

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
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
