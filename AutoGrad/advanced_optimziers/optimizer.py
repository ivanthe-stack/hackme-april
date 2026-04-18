from AutoGrad import Value
from NeuralNetwork import Network

class GD_Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def cross_enthropy_with_intiger_labels(self, p, label):
        sum = Value(0)
        for i in range(len(p)):
            sum += Value(-int(i == label)) * p[i].log()
        return sum

    def loss(self, network, inputs, labels):
        output = network.forward(inputs)
        print(f"Output: {output}")
        loss = self.cross_enthropy_with_intiger_labels(output, labels)
        print(f"Loss: {loss}")
        return loss

    def update_weights(self,network,loss):
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
