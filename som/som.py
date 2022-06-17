from .neuron import Neuron
import numpy as np


class Som:
    def __init__(self, layer_size = [4, 4], epoch = 4, sigma = 3, eta = 0.001, W = []):
        self.__layer = [
            [
                Neuron()
                for column in range(layer_size[1])
            ]
            for line in range(layer_size[0])
        ]

        self.__set_neurons_neighborhood()

        self.__epoch = epoch
        self.__sigma = sigma
        self.__eta = eta
        self.__W = W   


    def __set_neurons_neighborhood(self):
        for line_index, line in enumerate(self.__layer):
            for column_index, _ in enumerate(line):
                if len(self.__layer[line_index]) > 1:
                    # casos em que os valores só tem vizinho a esquerda ou a direita
                    if column_index == 0:
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index][column_index + 1])
                    elif column_index == len(self.__layer[line_index]) - 1:
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index][column_index - 1])
                    # caso em que os valores têm vizinhos a esquerda e a direita
                    else:
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index][column_index - 1])
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index][column_index + 1])
                
                if len(self.__layer) > 1:
                    # casos em que os valores só tem vizinho superior ou inferior
                    if line_index == 0:
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index + 1][column_index])
                    elif line_index == len(self.__layer) - 1:
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index - 1][column_index])
                    # caso em que os valores têm vizinhos superior e inferior
                    else:
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index - 1][column_index])
                        self.__layer[line_index][column_index].set_neighborhood(self.__layer[line_index + 1][column_index])


    def fit(self, X):
        self.__initializing_weights_of_all_neurons(X.shape[1])
        
        for _ in range(self.__epoch):
            for x in X.values:
                self.__update_similarity_values_of_all_neurons(x)

                winner = self.get_winner()

                self.__update_winner_neuron_neighborhood_similarity_values(winner)

                self.__update_weights_of_all_neurons(x)


    def __initializing_weights_of_all_neurons(self, x_size):
        for line in self.__layer:
            for neuron in line:
                if self.__W == []:
                    W = np.random.uniform(-1, 1, x_size)
                else:
                    W = self.__W
                
                neuron.update_weights(W)


    def __update_similarity_values_of_all_neurons(self, x):
        for line in self.__layer:
            for neuron in line:
                # discrimination function
                new_similarity_value = sum(np.power(x - neuron.get_weights(), 2))

                neuron.update_similarity_value(new_similarity_value)


    def get_winner(self):
        winner = self.__layer[0][0]
        for line in self.__layer:
            for neuron in line:
                if neuron.get_similarity_value() < winner.get_similarity_value():
                    winner = neuron

        return winner


    def __update_winner_neuron_neighborhood_similarity_values(self, winner):
        for neighbor in winner.get_neighborhoods():
            # topological neighborhood function
            new_similarity_value = np.exp(
                (winner.get_similarity_value() - neighbor.get_similarity_value()) / 2 * (self.__sigma**2)
            )

            neighbor.update_similarity_value(new_similarity_value)

    def __update_weights_of_all_neurons(self, x):
        for line in self.__layer:
            for neuron in line:
                # update weights function
                W = self.__eta * neuron.get_weights() * (x - neuron.get_old_similarity_value())

                neuron.update_weights(W)


    def show_layer_similarity_values(self):
        for il, line in enumerate(self.__layer):
            for ic, neuron in enumerate(line):
                format_value = ''

                if neuron == self.get_winner():
                    format_value += '(('

                format_value += '{0:.4f}'.format(self.__layer[il][ic].get_similarity_value())

                if neuron == self.get_winner():
                    format_value += '))'

                print(format_value, end = ' ')
            print()

