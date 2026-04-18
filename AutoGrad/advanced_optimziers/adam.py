from AutoGrad import Value
from NeuralNetwork import Network
from sgd import Data, SGD

class Adam(SGD):
    def __init__(self, learning_rate=0.01, epsilon=1e-8, momentum_decay=0.9, second_moment_decay=0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.momentum_decay = momentum_decay
        self.second_moment_decay = second_moment_decay
        self.grad_sum = {}
        self.grad_squared_sum = {}
        self.t = 0

    def update_grad_sum(self, weight, grad):
        if weight not in self.grad_sum:
            self.grad_sum[weight] = 0
        self.grad_sum[weight] = self.momentum_decay * self.grad_sum[weight] + (1 - self.momentum_decay) * grad

    def update_grad_squared_sum(self, weight, grad):
        if weight not in self.grad_squared_sum:
            self.grad_squared_sum[weight] = 0
        self.grad_squared_sum[weight] = self.second_moment_decay * self.grad_squared_sum[weight] + (1 - self.second_moment_decay) * (grad ** 2)

    def update_weights(self,network,loss):
        self.t += 1
        loss.backward()
        for layer in network.layers:
            for row in layer:
                for weight in row:
                    self.update_grad_sum(weight, weight.grad)
                    self.update_grad_squared_sum(weight, weight.grad)

                    m_hat = self.grad_sum[weight] / (1 - self.momentum_decay ** self.t)
                    v_hat = self.grad_squared_sum[weight] / (1 - self.second_moment_decay ** self.t)

                    weight.value -= self.learning_rate * m_hat / ((v_hat ** 0.5) + self.epsilon)

        return network


if __name__ == "__main__":
    optimizer = Adam(learning_rate=0.01)

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

"""
Adam workings:
Symbols
- theta - model parameter (one weight)
- g_t - current gradient at step t for that parameter
- m_t - first moment (moving average of gradients)
- v_t - second moment (moving average of squared gradients)
- beta1 - decay for m (usually 0.9)
- beta2 - decay for v (usually 0.999)
- eps - small constant for numerical stability (usually 1e-8)
- lr - learning rate (usually 0.001 as default Adam start)
- t - step counter (starts at 1, increases every optimizer step)
- m_hat_t, v_hat_t - bias-corrected versions of m_t, v_t
Initialization (per parameter)
- m_0 = 0
- v_0 = 0
Initialization (global optimizer state)
- t = 0
- choose constants:
  - lr = 0.001
  - beta1 = 0.9
  - beta2 = 0.999
  - eps = 1e-8
Update at each training step
1) Increase step counter
- t = t + 1
2) Compute gradient
- g_t = gradient of loss w.r.t current theta
3) Update first moment (mean-like)
- m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
4) Update second moment (variance-like)
- v_t = beta2 * v_(t-1) + (1 - beta2) * (g_t * g_t)
5) Bias correction (important in early steps)
- m_hat_t = m_t / (1 - beta1^t)
- v_hat_t = v_t / (1 - beta2^t)
6) Parameter update
- theta = theta - lr * m_hat_t / (sqrt(v_hat_t) + eps)
What is constant vs changing
- Constants: lr, beta1, beta2, eps
- Global changing value: t
- Per-parameter changing values: m_t, v_t, g_t, theta
Why bias correction exists
- At the start, m and v are initialized at 0, so they are biased toward 0 in first steps.
- m_hat and v_hat remove that startup bias.
- Without correction, first updates are often too small or mis-scaled.
"""
