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


    def __y_separate(self, y):
        self.__ndarray_classes = y.unique() # array com as classes existentes
        list_y_separate = []
        for each_class in self.__ndarray_classes:
            ndarray_index_class_zeros = np.where(self.__ndarray_classes != each_class)[0] # classe que receberá 1
            list_class_zeros = [self.__ndarray_classes[i] for i in ndarray_index_class_zeros] # classes que receberão 0

            new_y = y.replace(each_class, 1)
            new_y = new_y.replace(list_class_zeros, 0)
            list_y_separate.append(new_y) # adicionando o novo y para classificar a classe "each_classe"
        
        return list_y_separate


    def predict(self, x):
        predicts = []
        for perceptron in self.__adalines:
            predicts.append(perceptron.predict(x))


        # essa parte depende do tipo de problema, caso seja de classificação, 
        # então multiplica-se a lista predict com o ndarray de classes, caso
        # seja de regressão, então deve-se fazer uma margem, exemplo: predict <
        # 1.5 -> x, predict >= 1.5 -> y, ...
        index_predict = np.where((self.__ndarray_classes * predicts) != '')[0]

        # caso todas as predições sejam 0 ou todas as predições sejam 1, 
        # então a primeira classe será escolhida (poderia randomizar)
        if len(index_predict) == 0 or len(index_predict) != 1:
            index_predict = 0
        else:
            index_predict = index_predict[0]

        return self.__ndarray_classes[index_predict]


    def score(self, X_test, y_test):
        total_hits = 0
        for x, y in zip(X_test.values, y_test.values):
            predict = self.predict(x)
            if predict == y:
                total_hits += 1

        return total_hits/y_test.size
