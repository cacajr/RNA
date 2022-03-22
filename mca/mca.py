from adaline.adaline import Adaline
import numpy as np


class MCA:
    def __init__(self, num_class = 3, bies = -1, eta = 0.001, epoch = 100, W = [], activation_function = 'hardlim'):
        self.__adalines = [Adaline(bies, eta, epoch, W, activation_function) for _ in range(num_class)]
        self.__ndarray_classes = np.ndarray([])


    def get_perceptrons(self):
        return self.__adalines


    def get_weights(self):
        return [adaline.get_weight() for adaline in self.__adalines]


    def fit(self, X, y):
        # divisão das classes para treinar cada uma individualmente
        list_y_separate = self.__y_separate(y)

        # treinando cada classe com um adaline distinto
        for adaline, y_separate in zip(self.__adalines, list_y_separate):
            adaline.fit(X, y_separate)

            # print(adaline.score(X, y_separate))


    def __y_separate(self, y):
        self.__ndarray_classes = y.unique() # array com as classes existentes
        list_y_separate = []
        for each_class in self.__ndarray_classes:
            ndarray_index_class_zeros = np.where(self.__ndarray_classes != each_class)[0] # index das classes que receberão 0
            list_class_zeros = [self.__ndarray_classes[i] for i in ndarray_index_class_zeros] # classes que receberão 0

            new_y = y.replace(each_class, 1)
            new_y = new_y.replace(list_class_zeros, [0 for _ in range(len(list_class_zeros))])
            list_y_separate.append(new_y) # adicionando o novo y para classificar a classe "each_classe"
        
        return list_y_separate


    def predict(self, x):
        predicts = []
        for adaline in self.__adalines:
            predicts.append(adaline.predict(x))

        predict_activation = [1 if max(predicts) == predict else 0 for predict in predicts]
        predict_index = np.where((self.__ndarray_classes * predict_activation) != '')[0][0]

        return self.__ndarray_classes[predict_index]


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
