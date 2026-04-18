from AutoGrad import Value
from NeuralNetwork import Network
from sgd import SGD, Data
import copy

class SGD_momentum(SGD):
    def __init__(self, learning_rate=0.01, momentum_decay=0.9):
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.velocity = {}

    def update_velocity(self, weight, grad):
        if weight not in self.velocity:
            self.velocity[weight] = 0
        self.velocity[weight] = self.momentum_decay * self.velocity[weight] + grad * (1 - self.momentum_decay)

    def update_weights(self,network,loss):
        loss.backward()
        for layer in network.layers:
            for row in layer:
                for weight in row:
                    self.update_velocity(weight, weight.grad)
                    weight.value -= self.learning_rate * self.velocity[weight]
        return network

if __name__ == "__main__":
    optimizer = SGD(learning_rate=0.01)
    optimizer_momentum = SGD_momentum(learning_rate=0.01)

    network = Network([2, 3, 2])
    network_momentum = copy.deepcopy(network)

    inputs = []
    labels = []
    for i in range(100):
        x1 = Value(1 if i % 2 == 0 else 0)
        x2 = Value(1 if i % 3 == 0 else 0)
        inputs.append([x1, x2])
        labels.append(0 if (i % 6 != 0 or i %5 != 0) else 1)
    data = Data(inputs, labels)
    batch = data.feed(batch_size=10)

    for i in range(10):
        print("\nBackproping..\n")
        step_inputs, step_label = next(batch)

        loss = optimizer.loss(network, step_inputs, step_label)
        network = optimizer.update_weights(network, loss)
        loss.zero_grads()
        network.print(1)

        loss_momentum = optimizer_momentum.loss(network_momentum, step_inputs, step_label)
        network_momentum = optimizer_momentum.update_weights(network_momentum, loss_momentum)
        loss_momentum.zero_grads()
        network_momentum.print(1)
