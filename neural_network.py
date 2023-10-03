
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)
    
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_pass(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples= len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_pass(output)

                err += self.loss(y_train[j],output)

                error = self.loss_prime(y_train[j],output)
                for layer in reversed(self.layers):
                    error = layer.backward_pass(error, learning_rate)

            err /= samples
            print('Época -> {}/{} Error={}'.format(i+1,epochs,err))
        print('Real data: {}\nPredicted: {}'.format(y_train,output))
