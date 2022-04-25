import numpy as np
import math


class PerceptronMMQ:
    def __init__(self, bies = -1, eta = 0.001, W = [], activation_function = 'hardlim'):
        self.__bies = bies
        self.__eta = eta
        self.__W = W
        self.__activation_function = activation_function
    

    def get_bies(self):
        return self.__bies


    def get_weight(self):
        return self.__W


    def set_weight(self, W):
        self.__W = W


    def fit(self, X, y):
        # averiguando se veio pesos definidos no parâmetro
        if self.__W == []:
            # array de pesos aleatórios das entradas com o peso do bies no índice 0
            self.__W = np.random.uniform(-1, 1, X.shape[1] + 1)
        elif len(self.__W) != X.shape[1] + 1:
            print('Array de pesos incompatível com entrada!')
            return None

        pass

                
    def activation_function(self, u):
        if self.__activation_function == 'hardlim':
            return 0 if u < 0 else 1
        if self.__activation_function == 'relu':
            return 0 if u <= 0 else 1
        if self.__activation_function == 'sign':
            return -1 if u <= 0 else 1
        if self.__activation_function == 'sigmoid':
            return 1/(1 + np.exp(-u))
        if self.__activation_function == 'tanh':
            return (math.exp(2 * u) - 1) / (math.exp(2 * u) + 1)


    def predict(self, x, activation_function = True):
        
        pass

        # função de ativação
        return self.activation_function(u) if activation_function else u


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
