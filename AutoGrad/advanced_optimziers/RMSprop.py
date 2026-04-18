from AutoGrad import Value
from NeuralNetwork import Network
from sgd import Data
from adagrad import Adagrad

class RMSprop(Adagrad):
    def __init__(self, learning_rate=0.01, epsilon=1e-8, momentum_decay=0.9):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.momentum_decay = momentum_decay
        self.grad_squared_sum = {}

    def update_grad_squared_sum(self, weight, grad):
        if weight not in self.grad_squared_sum:
            self.grad_squared_sum[weight] = 0
        self.grad_squared_sum[weight] *= self.momentum_decay
        self.grad_squared_sum[weight] += grad ** 2 * (1 - self.momentum_decay)

if __name__ == "__main__":
    optimizer = RMSprop(learning_rate=0.01)

    network = Network([2, 3, 2])
    inputs = []
    labels = []
    for i in range(100):
        x1 = Value(1 if i % 2 == 0 else 0)
        x2 = Value(1 if i % 3 == 0 else 0)
        inputs.append([x1, x2])
        labels.append(0 if (i % 4 != 0 or i %5 != 0) else 1)
    data = Data(inputs, labels)
    batch = data.feed(batch_size=10)

    for i in range(10):
        print("\nBackproping..\n")
        loss = optimizer.loss(network, *next(batch))
        network = optimizer.update_weights(network, loss)
        loss.zero_grads()
        network.print(1)
