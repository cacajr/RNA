import numpy as np
import math


class Perceptron:
    def __init__(self, bies = -1, eta = 0.001, epoch = 100, W = [], activation_function = 'hardlim'):
        self.__bies = bies
        self.__eta = eta
        self.__epoch = epoch
        self.__W = W
        self.__activation_function = activation_function
        self.__err = 0 # usado apenas no MLP
    

    def get_bies(self):
        return self.__bies


    def get_weight(self):
        return self.__W


    def set_weight(self, W):
        self.__W = W


    def get_error(self):
        return self.__err


    def set_error(self, err):
        self.__err = err


    def fit(self, X, y, learn_rule = True):
        # averiguando se veio pesos definidos no parâmetro
        if self.__W == []:
            # array de pesos aleatórios das entradas com o peso do bies no índice 0
            self.__W = np.random.uniform(-1, 1, X.shape[1] + 1)
        elif len(self.__W) != X.shape[1] + 1:
            print('Array de pesos incompatível com entrada!')
            return None

        for _ in range(self.__epoch):
            for x, yn in zip(X.values, y.values):
                x = np.hstack((self.__bies, x)) # concatenando o bies

                # produto interno entre as amostras x com bies e os pesos W
                u = np.dot(x, self.__W)
                
                # função de ativação
                pred = self.activation_function(u)

                # se errou, aplica-se a regra de aprendizado
                if pred != yn and learn_rule:
                    e = yn - pred
                    self.__W += self.__eta * (e * x)

    def fit(self, x, yn, activation_function = True): # fit usado para o MLP
        # averiguando se veio pesos definidos no parâmetro
        if self.__W == []:
            # array de pesos aleatórios das entradas com o peso do bies no índice 0
            self.__W = np.random.uniform(-1, 1, len(x) + 1)

        x = np.hstack((self.__bies, x)) # concatenando o bies

        # produto interno entre as amostras x com bies e os pesos W
        u = np.dot(x, self.__W)
        
        # função de ativação
        pred = self.activation_function(u) if activation_function else u

        # guardando o erro
        self.__err = yn - self.activation_function(u)

        return pred

                
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
        # produto interno das amostras x com bies e os pesos W
        x = np.hstack((self.__bies, x))
        u = np.dot(x, self.__W)

        # função de ativação
        return self.activation_function(u) if activation_function else u


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
