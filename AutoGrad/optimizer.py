from AutoGrad import Value
from NeuralNetwork import Network

class GD_Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate # speed of learning, often between 0.01 and 0.0001

    def cross_enthropy_with_intiger_labels(self, p, label): # Actual loss function, not really important
        sum = Value(0)
        for i in range(len(p)):
            sum += Value(-int(i == label)) * p[i].log()
        return sum

    def loss(self, network, inputs, labels): # helper function
        output = network.forward(inputs)
        loss = self.cross_enthropy_with_intiger_labels(output, labels)
        return loss

    def update_weights(self,network,loss): # gradient descent step
        loss.backward()
        for layer in network.layers:
            for row in layer:
                for weight in row:
                    weight.value -= self.learning_rate * weight.grad
        return network

if __name__ == "__main__":

    Eva = GD_Optimizer(learning_rate=0.01)

    network = Network([2, 3, 2])
    inputs = [Value(1), Value(2)]
    label = 0

    for i in range(10):
        print("\nBackproping..\n")
        loss = Eva.loss(network, inputs, label)
        network = Eva.update_weights(network, loss)
        loss.zero_grads()
        network.print(1)
