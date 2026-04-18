import numpy as np
import time
import os
import argparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--load', default=None, help='load model from file')
parser.add_argument('-n', '--name', default='model.npz', help='save model to file')
parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
parser.add_argument('-p', '--print_every', type=int, default=1000, help='print every n epochs')
parser.add_argument('-c', '--csv', default=None, help='save training log to csv')
parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('-o', '--optimizer', default='sgd', choices=['sgd', 'adam'], help='optimizer')
args = parser.parse_args()

LAYERS = [64, 16, 10]
lr = args.lr
optimizer = args.optimizer
epochs = args.epochs
print_every = args.print_every
seed = args.seed
clip_grad = 1.0

np.random.seed(seed)

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def init_adam(self):
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, x):
        self.xs = [x]
        for i in range(len(self.weights)):
            x = x @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)
            self.xs.append(x)
        return np.exp(x - x.max(axis=1, keepdims=True))

    def train(self, X, y, X_test, y_test, epochs, lr, print_every=1):
        print(f"seed={seed}")
        print(f"arch={LAYERS}")
        print(f"lr={lr}")
        print(f"{'Epoch':<6} {'TrainLoss':<12} {'TestLoss':<12} {'TestAcc':<10} {'Time':<10} {'Time/Epoch':<10}")
        self.losses = []
        self.csv_data = []
        cum_time = 0
        
        if optimizer == 'adam':
            self.init_adam()
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for epoch in range(epochs):
            t0 = time.perf_counter()
            out = self.forward(X)
            loss = -np.mean(np.log(out[np.arange(len(y)), y] + 1e-8))
            self.losses.append(loss)
            dout = out.copy()
            dout[np.arange(len(y)), y] -= 1
            for i in range(len(self.weights) - 1, -1, -1):
                dw = self.xs[i].T @ dout / len(y)
                db = np.sum(dout, axis=0) / len(y)
                dw = np.clip(dw, -clip_grad, clip_grad)
                db = np.clip(db, -clip_grad, clip_grad)
                
                if optimizer == 'adam':
                    self.t += 1
                    self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw
                    self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dw ** 2)
                    self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db
                    self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db ** 2)
                    m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
                    v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
                    m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
                    v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
                    self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
                    self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
                else:
                    self.weights[i] -= lr * dw
                    self.biases[i] -= lr * db
                
                if i > 0:
                    dout = dout @ self.weights[i].T * (self.xs[i] > 0)
            cum_time += time.perf_counter() - t0
            pred = np.argmax(self.forward(X_test), axis=1)
            acc = np.mean(pred == y_test)
            test_out = self.forward(X_test)
            test_loss = -np.mean(np.log(test_out[np.arange(len(y_test)), y_test] + 1e-8))
            self.csv_data.append({'epoch': epoch, 'train_loss': loss, 'test_loss': test_loss, 'test_acc': acc, 'time': cum_time})
            if epoch % print_every == 0:
                print(f"{epoch:<6} {loss:<12.4f} {test_loss:<12.4f} {acc:<10.4f} {cum_time:<10.4f} {cum_time/print_every:<10.4f}")
                cum_time = 0

    def save(self, path):
        np.savez(path, weights=np.array(self.weights, dtype=object), biases=np.array(self.biases, dtype=object), layers=np.array(self.layers))

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        self.layers = list(data['layers'])

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=seed)
X_train = X_train / 16.0
X_test = X_test / 16.0

mlp = MLP(LAYERS)
if args.load:
    mlp.load(args.load)
    print(f"Loaded model from {args.load}")
mlp.train(X_train, y_train, X_test, y_test, epochs, lr, print_every)
if args.csv:
    mlp.csv_data[0]['seed'] = seed
    mlp.csv_data[0]['arch'] = str(LAYERS)
    mlp.csv_data[0]['lr'] = lr
    mlp.csv_data[0]['clip_grad'] = clip_grad
    for row in mlp.csv_data:
        row.setdefault('seed', '')
        row.setdefault('arch', '')
        row.setdefault('lr', '')
        row.setdefault('clip_grad', '')
    import csv
    with open(args.csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'test_loss', 'test_acc', 'time', 'seed', 'arch', 'lr', 'clip_grad'])
        writer.writeheader()
        writer.writerows(mlp.csv_data)
    print(f"Saved {args.csv}")
mlp.save(args.name)
print(f"Saved {args.name}")
print(f"seed={seed}")
print(f"arch={LAYERS}")
print(f"epochs={epochs}")
print(f"lr={lr}")

import matplotlib.pyplot as plt
plt.plot(mlp.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss.png')
print("Saved loss.png")
