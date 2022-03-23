import numpy as np
import math


class Adaline:
    def __init__(self, bies = -1, eta = 0.001, epoch = 100, W = [], activation_function = 'hardlim'):
        self.__bies = bies
        self.__eta = eta
        self.__epoch = epoch
        self.__W = W
        self.__activation_function = activation_function
    
    
    def get_bies(self):
        return self.__bies


    def get_weight(self):
        return self.__W


    def fit(self, X, y):
        # averiguando se veio pesos definidos no parâmetro
        if self.__W == []:
            # array de pesos aleatórios das entradas com o peso do bies no índice 0
            self.__W = np.random.uniform(-1, 1, X.shape[1] + 1) # estourando o ndarray() | !solução linha 41
        elif len(self.__W) != X.shape[1] + 1:
            print('Array de pesos incompatível com entrada!')
            return None

        for _ in range(self.__epoch):
            for x, yn in zip(X.values, y.values):
                x = np.hstack((self.__bies, x)) # concatenando o bies

                # produto interno entre as amostras x com bies e os pesos W
                pred = np.dot(x, self.__W)

                # armazeno o quanto o modelo errou para em seguida ajustá-lo
                e = yn - pred
                self.__W += self.__eta * (e * x)
                # self.__W = np.add(self.__W, self.__eta * (e * x), out=self.__W, casting='unsafe')
                # np.seterr(all='ignore', over='raise') # solução para o estouro, aumenta o formato e ignora o warn


    def activation_function(self, u):
        if self.__activation_function == 'hardlim':
            return 0 if u < 0 else 1
        if self.__activation_function == 'relu':
            return 0 if u <= 0 else 1
        if self.__activation_function == 'sign':
            return -1 if u <= 0 else 1
        if self.__activation_function == 'tanh':
            return (math.exp(2 * u) - 1) / (math.exp(2 * u) + 1)


    def predict(self, x, activation_function = True):
        # produto interno entre as amostras x com bies e os pesos W
        x = np.hstack((self.__bies, x))
        pred = np.dot(x, self.__W)

        # função de ativação
        return self.activation_function(pred) if activation_function else pred


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
