class Neuron:
    def __init__(self, similarity_value = 0):
        self.__similarity_value = similarity_value
        self.__old_similarity_value = None
        self.__W = []
        self.__neighborhood = []


    def get_similarity_value(self):
        return self.__similarity_value

    
    def get_old_similarity_value(self):
        return self.__old_similarity_value


    def get_neighborhoods(self):
        return self.__neighborhood


    def get_weights(self):
        return self.__W


    def set_neighborhood(self, neuron):
        self.__neighborhood.append(neuron)


    def update_weights(self, W):
        if self.__W == []:
            self.__W = W
        else:
            self.__W += W


    def update_similarity_value(self, new_similarity_value):
        # if old value is None, so is first value
        if self.__old_similarity_value is None:
            self.__old_similarity_value = new_similarity_value
        else:
            self.__old_similarity_value = self.__similarity_value
        
        self.__similarity_value = new_similarity_value