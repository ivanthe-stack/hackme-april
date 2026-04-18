from AutoGrad import Value
from NeuralNetwork import Network
from sgd import Data
from adam import Adam

class AdamW(Adam):
    def __init__(self, learning_rate=0.01, epsilon=1e-8, momentum_decay=0.9, second_moment_decay=0.999, weight_decay=0.01):
        super().__init__(learning_rate, epsilon, momentum_decay, second_moment_decay)
        self.weight_decay = weight_decay

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

                    # Decoypled weight decay step:
                    weight.value -= self.learning_rate * self.weight_decay * weight.value

        return network


if __name__ == "__main__":
    optimizer = AdamW(learning_rate=0.01)

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
AdamW workings:
Symbols
- theta - model parameter (one weight)
- g_t - current gradient at step t for that parameter
- m_t - first moment (moving average of gradients)
- v_t - second moment (moving average of squared gradients)
- beta1 - decay for m (usually 0.9)
- beta2 - decay for v (usually 0.999)
- eps - small constant for numerical stability (usually 1e-8)
- lr - learning rate (usually 0.001)
- wd - weight decay coefficient (usually 0.01)
- t - step counter

Initialization
- m_0 = 0
- v_0 = 0
- t = 0

Update at each training step
1) t = t + 1
2) g_t = gradient of loss w.r.t theta (no L2 term added to g_t)
3) m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
4) v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
5) m_hat_t = m_t / (1 - beta1^t)
6) v_hat_t = v_t / (1 - beta2^t)
7) Adam step:
   theta = theta - lr * m_hat_t / (sqrt(v_hat_t) + eps)
8) Decoupled weight decay step:
   theta = theta - lr * wd * theta

Equivalent one-line update:
theta = theta - lr * (m_hat_t / (sqrt(v_hat_t) + eps) + wd * theta)
"""
