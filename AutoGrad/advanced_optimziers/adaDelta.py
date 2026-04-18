from AutoGrad import Value
from NeuralNetwork import Network
from sgd import Data
from RMSprop import RMSprop

class AdaDelta(RMSprop):
    def __init__(self, epsilon=1e-8, momentum_decay=0.9):
        self.epsilon = epsilon
        self.momentum_decay = momentum_decay
        self.grad_squared_sum = {}
        self.delta_squared_sum = {}

    def update_delta_squared_sum(self, weight, delta):
        if weight not in self.delta_squared_sum:
            self.delta_squared_sum[weight] = 0
        self.delta_squared_sum[weight] *= self.momentum_decay
        self.delta_squared_sum[weight] += delta ** 2 * (1 - self.momentum_decay)

    def update_weights(self,network,loss):
        loss.backward()
        for layer in network.layers:
            for row in layer:
                for weight in row:
                    self.update_grad_squared_sum(weight, weight.grad)
                    RMS_g = (self.grad_squared_sum[weight] + self.epsilon) ** 0.5
                    try:
                        RMS_dx = (self.delta_squared_sum[weight] + self.epsilon) ** 0.5
                    except KeyError:
                        RMS_dx = self.epsilon ** 0.5
                    delta = (RMS_dx / RMS_g) * weight.grad
                    weight.value -= delta
                    self.update_delta_squared_sum(weight, delta)
        return network


if __name__ == "__main__":
    optimizer = AdaDelta()

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
AdaDelta workings:

Symbols
- theta - model parameter (a weight value you are optimizing)
- g_t - current gradient at step t for that parameter (dLoss/dtheta)
- rho - decay factor for moving averages (usually 0.9 or 0.95)
- eps - tiny constant to avoid divide-by-zero (usually 1e-6)
- E_g2_t - running average of squared gradients up to step t
- E_dx2_t - running average of squared parameter updates up to step t
- dx_t - actual update applied to parameter at step t
Think of E_g2 and E_dx2 as optimizer memory stored per weight.
Initialization (per parameter)
- E_g2_0 = 0
- E_dx2_0 = 0
- choose constants once:
  - rho = 0.9 (common)
  - eps = 1e-6 (common)
Update at each training step
1) Compute gradient
- g_t = gradient of loss w.r.t current theta
2) Update gradient history
- E_g2_t = rho * E_g2_(t-1) + (1 - rho) * (g_t * g_t)
3) Build RMS terms
- RMS_g_t = sqrt(E_g2_t + eps)
- RMS_dx_(t-1) = sqrt(E_dx2_(t-1) + eps)
4) Compute parameter update
- dx_t = - (RMS_dx_(t-1) / RMS_g_t) * g_t
5) Apply update
- theta = theta + dx_t
6) Update update-history
- E_dx2_t = rho * E_dx2_(t-1) + (1 - rho) * (dx_t * dx_t)
"""
