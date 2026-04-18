from AutoGrad import Value
import random, math

class Network:

    def random_weight(self, input_size=1):
        return Value(random.uniform(-math.sqrt(6/input_size), math.sqrt(6/input_size)))

    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = []
            for _ in range(layer_sizes[i+1]):
                rows = []
                for _ in range(layer_sizes[i]):
                    rows.append(self.random_weight(layer_sizes[i]))
                layer.append(rows)
            self.layers.append(layer)
        self.print()

    def forward(self, x):
        for layer_index in range(len(self.layers)):
            y = []
            for i in range(len(self.layers[layer_index])):
                sum = Value(0)
                for j in range(len(self.layers[layer_index][i])):
                    result = self.layers[layer_index][i][j] * x[j]
                    sum = sum + result
                y.append(sum)
            if layer_index < len(self.layers) - 1:
                x = []
                for i in range(len(y)):
                    x.append(y[i].relu())
            else:
                x = y
        x = self.softmax(x)
        return x

    def print(self,max_depth=3):
        for layer in self.layers:
            print(f"Layer:")
            for row in layer:
                print(f"  Neuron:")
                for weight in range(len(row)):
                    if weight >= max_depth:
                        print(f"    ...")
                        break
                    print(f"    Weight: {row[weight]}")

    def softmax(self, x):
        sum = Value(0)
        for i in range(len(x)):
            sum += math.exp(1)**x[i]
        for i in range(len(x)):
            x[i] = (math.exp(1)**x[i])
            x[i] = x[i] / sum
        return x
