import sys
sys.path.insert(0, 'AutoGrad')
from NeuralNetwork import Network
from AutoGrad import Value
from optimizer import GD_Optimizer
import random, time

LAYERS = [64, 16, 10]
lr = 0.01
epochs = 10
nn = Network(LAYERS)
opt = GD_Optimizer(lr)

X_train = [[Value(random.uniform(0, 1)) for _ in range(64)] for _ in range(100)]
y_train = [random.randint(0, 9) for _ in range(100)]

def train(epochs):
    print(f"lr={lr}")
    print(f"{'Epoch':<6} {'Loss':<10} {'Time':<8}")
    losses = []
    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0
        for x, y in zip(X_train, y_train):
            out = nn.forward(x)
            loss = -out[y].log()
            total_loss += loss.value
            loss.backward()
            opt.update_weights(nn, loss)
            loss.zero_grads()
        losses.append(total_loss/len(X_train))
        print(f"{epoch:<6} {losses[-1]:<10.4f} {time.time()-t0:<8.2f}")
    return losses

if __name__ == "__main__":
    losses = train(epochs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png')
    print("Saved loss.png")