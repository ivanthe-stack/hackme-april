from AutoGrad import Value
from NeuralNetwork import Network
from sgd import SGD, Data

class Adagrad(SGD):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.grad_squared_sum = {}

    def update_grad_squared_sum(self, weight, grad):
        if weight not in self.grad_squared_sum:
            self.grad_squared_sum[weight] = 0
        self.grad_squared_sum[weight] += grad ** 2

    def update_weights(self,network,loss):
        loss.backward()
        for layer in network.layers:
            for row in layer:
                for weight in row:
                    self.update_grad_squared_sum(weight, weight.grad)
                    adjusted_lr = self.learning_rate / (self.grad_squared_sum[weight] ** 0.5 + self.epsilon)
                    weight.value -= adjusted_lr * weight.grad
        return network

if __name__ == "__main__":
    optimizer = Adagrad(learning_rate=0.01)

    network = Network([2, 3, 2])
    inputs = []
    labels = []
    for i in range(100):
        x1 = Value(1 if i % 2 == 0 else 0)
        x2 = Value(1 if i % 3 == 0 else 0)
        inputs.append([x1, x2])
        labels.append(0 if (i % 4 != 0 and i %5 != 0) else 1)
    data = Data(inputs, labels)
    batch = data.feed(batch_size=10)

    for i in range(10):
        print("\nBackproping..\n")
        loss = optimizer.loss(network, *next(batch))
        network = optimizer.update_weights(network, loss)
        loss.zero_grads()
        network.print(1)
