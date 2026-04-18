from AutoGrad import Value
from NeuralNetwork import Network
from optimizer import GD_Optimizer

class Data:
    def __init__(self, inputs,labels):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
        return len(self.inputs)
    def feed(self,batch_size):
        batches = []
        for i in range(0, len(self.labels), batch_size):
            batch_inputs = self.inputs[i:i+batch_size]
            batch_labels = self.labels[i:i+batch_size]
            batches.append((batch_inputs, batch_labels))
        return iter(batches)

class SGD(GD_Optimizer):
    def loss(self, network, inputs, labels):  # BATCH LOSS
        loss = Value(0)
        print(f"Calculating loss on {len(inputs)} samples...")
        for i in range(len(labels)):
            output = network.forward(inputs[i])
            loss += self.cross_enthropy_with_intiger_labels(output, labels[i])
        loss /= Value(len(labels))
        print(f"Loss: {loss.value}")
        return loss

if __name__ == "__main__":

    optimizer = SGD(learning_rate=0.01)

    network = Network([2, 3, 2])
    inputs = []
    labels = []
    for i in range(100):
        x1 = Value(1 if i % 2 == 0 else 0)
        x2 = Value(1 if i % 3 == 0 else 0)
        inputs.append([x1, x2])
        labels.append(0 if (i % 2 == 0 and i % 3 == 0) else (1 if i % 2 == 0 else 2))
    data = Data(inputs, labels)
    batch = data.feed(batch_size=10)

    for i in range(10):
        print("\nBackproping..\n")
        loss = optimizer.loss(network, *next(batch))
        network = optimizer.update_weights(network, loss)
        loss.zero_grads()
        network.print(1)
