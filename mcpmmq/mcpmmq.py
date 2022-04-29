import numpy as np
import math
from perceptronmmq.perceptronmmq import PerceptronMMQ


class MCPMMQ:
    def __init__(self, num_class = 3, bies = -1, activation_function = 'hardlim'):
        self.__bies = bies
        self.__perceptronsmmq = [PerceptronMMQ(bies, activation_function) for _ in range(num_class)]
        self.__ndarray_classes = np.ndarray([])


    def get_bies(self):
        return self.__bies


    def get_weight(self):
        return [perceptronmmq.get_weight() for perceptronmmq in self.__perceptronsmmq]


    def fit(self, X, y):
        # divisão das classes para treinar cada uma individualmente
        list_y_separate = self.__y_separate(y)

        # treinando cada classe com um perceptron distinto
        for perceptronmmq, y_separate in zip(self.__perceptronsmmq, list_y_separate):
            perceptronmmq.fit(X, y_separate)

                
    def __y_separate(self, y):
        list_y_separate = []

        self.__ndarray_classes = y.unique()
        one_hot_aux = np.zeros(self.__ndarray_classes.size, dtype=int)
        one_hot = np.ones(self.__ndarray_classes.size, dtype=int)

        for index, _ in enumerate(self.__ndarray_classes):
            one_hot_aux[index] = 1
            list_y_separate.append(y.replace(self.__ndarray_classes, (one_hot * one_hot_aux)))
            one_hot_aux[index] = 0

        return list_y_separate


    def predict(self, x):
        predicts = []
        for perceptronmmq in self.__perceptronsmmq:
            predicts.append(perceptronmmq.predict(x, False)) # retiro a função de ativação do predict

        biggest_predict = max(predicts)

        one_hot = np.array([1 if predict == biggest_predict else 0 for predict in predicts])
        predict_index = np.where((self.__ndarray_classes * one_hot) != '')[0][0]

        return self.__ndarray_classes[predict_index]


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
